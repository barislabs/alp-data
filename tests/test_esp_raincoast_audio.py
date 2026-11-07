"""Offline regressions for Raincoast channel selection and time-axis resampling."""

import csv
import math
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from alp_data.dataset import dataset_from_config
from alp_data.datasets.esp_raincoast import ESPRaincoast, ESPRaincoastConfig

NATIVE_RATE = 8000
TOTAL_FRAMES = 4097
BEGIN_TIME = 0.125
END_TIME = 0.375
START_FRAME = 1000
END_FRAME = 3000
MONO_METHODS = [None, "average", "keep_first"]
SAMPLE_RATES = [None, NATIVE_RATE, 4000, 12000]


@pytest.fixture
def local_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Write lossless local audio and replace the manifest with a local CSV.

    Returns
    -------
    dict[str, np.ndarray]
        Original float32 samples keyed by the manifest's local audio filenames.
    """
    time = np.arange(TOTAL_FRAMES) / NATIVE_RATE
    left = (0.35 * np.sin(2 * np.pi * 233 * time) + 0.1).astype(np.float32)
    right = (0.2 * np.cos(2 * np.pi * 617 * time) - 0.15).astype(np.float32)
    mono = (0.25 * np.sin(2 * np.pi * 389 * time) + 0.05).astype(np.float32)
    sources = {"stereo.wav": np.column_stack((left, right)), "mono.wav": mono}

    manifest = tmp_path / "selection_table.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["local_path", "Begin Time (s)", "End Time (s)", "Sound Type"],
        )
        writer.writeheader()
        for name, audio in sources.items():
            sf.write(tmp_path / name, audio, NATIVE_RATE, subtype="FLOAT")
            writer.writerow(
                {
                    "local_path": name,
                    "Begin Time (s)": BEGIN_TIME,
                    "End Time (s)": END_TIME,
                    "Sound Type": "synthetic-call",
                }
            )

    monkeypatch.setattr(ESPRaincoast.info, "split_paths", {"full": str(manifest)})
    return sources


def _expected_audio(
    source: np.ndarray,
    mono_method: str | None,
    sample_rate: int | None,
    load_audio_segments: bool,
) -> np.ndarray:
    """Build the reference from fixture samples, resampling only 1D signals.

    Returns
    -------
    np.ndarray
        Expected samples after segment selection, channel selection and resampling.
    """
    audio = source[START_FRAME:END_FRAME] if load_audio_segments else source
    if audio.ndim == 2:
        if mono_method == "average":
            audio = (audio[:, 0] + audio[:, 1]) / 2
        elif mono_method == "keep_first":
            audio = audio[:, 0]

    if sample_rate is None or sample_rate == NATIVE_RATE:
        return audio

    channels = [audio] if audio.ndim == 1 else [audio[:, 0], audio[:, 1]]
    resampled = [
        librosa.resample(
            channel,
            orig_sr=NATIVE_RATE,
            target_sr=sample_rate,
            scale=True,
            res_type="kaiser_best",
        )
        for channel in channels
    ]
    return resampled[0] if audio.ndim == 1 else np.column_stack(resampled)


def _assert_audio(
    audio: np.ndarray,
    reported_rate: int,
    source: np.ndarray,
    mono_method: str | None,
    sample_rate: int | None,
    load_audio_segments: bool,
) -> None:
    """Check values and the independent frame, channel, and duration contract."""
    output_rate = NATIVE_RATE if sample_rate is None else sample_rate
    input_frames = END_FRAME - START_FRAME if load_audio_segments else TOTAL_FRAMES
    output_frames = math.ceil(input_frames * output_rate / NATIVE_RATE)
    expected_shape = (
        (output_frames, 2) if source.ndim == 2 and mono_method is None else (output_frames,)
    )
    assert reported_rate == output_rate
    assert audio.dtype == np.float32
    assert audio.shape == expected_shape
    assert audio.shape[0] / reported_rate == pytest.approx(
        input_frames / NATIVE_RATE, abs=1 / output_rate
    )
    expected = _expected_audio(source, mono_method, sample_rate, load_audio_segments)
    if output_rate == NATIVE_RATE:
        np.testing.assert_array_equal(audio, expected)
    else:
        np.testing.assert_allclose(audio, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("mono_method", MONO_METHODS, ids=["stereo", "average", "keep-first"])
@pytest.mark.parametrize("sample_rate", SAMPLE_RATES, ids=["native", "same", "down", "up"])
@pytest.mark.parametrize("load_audio_segments", [False, True], ids=["full", "segment"])
def test_audio_channels_and_sample_rate(
    local_audio: dict[str, np.ndarray],
    mono_method: str | None,
    sample_rate: int | None,
    load_audio_segments: bool,
) -> None:
    """All channel/rate choices work for stereo and mono files and segments."""
    dataset = ESPRaincoast(
        mono_method=mono_method,
        sample_rate=sample_rate,
        load_audio_segments=load_audio_segments,
    )
    assert len(dataset) == len(local_audio)
    for index, (name, source) in enumerate(local_audio.items()):
        sample = dataset[index]
        assert sample["local_path"] == name
        assert sample["Sound Type"] == "synthetic-call"
        assert sample["Begin Time (s)"] == BEGIN_TIME
        assert sample["End Time (s)"] == END_TIME
        _assert_audio(
            sample["audio"],
            sample["sample_rate"],
            source,
            mono_method,
            sample_rate,
            load_audio_segments,
        )


@pytest.mark.parametrize("from_config", [False, True], ids=["constructor", "config"])
def test_omitted_defaults_preserve_stereo(
    local_audio: dict[str, np.ndarray], from_config: bool
) -> None:
    """Omitting channel and rate settings preserves stereo at its native rate."""
    if from_config:
        config = ESPRaincoastConfig(data_root=None)
        assert config.mono_method is None
        assert config.sample_rate is None
        dataset, metadata = dataset_from_config(config)
        assert metadata == {}
    else:
        dataset = ESPRaincoast()

    assert dataset.mono_method is None
    assert dataset.sample_rate is None
    assert dataset.load_audio_segments is True
    sample = dataset[0]
    _assert_audio(
        sample["audio"], sample["sample_rate"], local_audio["stereo.wav"], None, None, True
    )
    assert sample["Begin Time (s)"] == BEGIN_TIME
    assert sample["End Time (s)"] == END_TIME


@pytest.mark.parametrize("mono_method", MONO_METHODS, ids=["stereo", "average", "keep-first"])
@pytest.mark.parametrize(
    "backend,streaming,sample_rate,load_audio_segments",
    [
        ("pandas", False, 4000, True),
        ("polars", False, 12000, False),
        ("pandas", True, 12000, False),
        ("polars", True, 4000, True),
    ],
    ids=["pandas-indexed", "polars-indexed", "pandas-streaming", "polars-streaming"],
)
def test_config_and_output_mapping_across_backends(
    local_audio: dict[str, np.ndarray],
    mono_method: str | None,
    backend: str,
    streaming: bool,
    sample_rate: int,
    load_audio_segments: bool,
) -> None:
    """Configuration carries channel/rate choices through both backend access paths."""
    mapping = {
        "audio": "waveform",
        "sample_rate": "rate",
        "Begin Time (s)": "begin",
        "End Time (s)": "end",
        "local_path": "path",
    }
    config = ESPRaincoastConfig(
        data_root=None,
        mono_method=mono_method,
        sample_rate=sample_rate,
        load_audio_segments=load_audio_segments,
        backend=backend,
        streaming=streaming,
        output_take_and_give=mapping,
    )
    dataset, metadata = dataset_from_config(config)
    assert metadata == {}
    assert dataset.mono_method == mono_method
    assert dataset.sample_rate == sample_rate
    assert dataset.load_audio_segments == load_audio_segments
    samples = [sample for sample in dataset] if streaming else [dataset[0], dataset[1]]
    assert len(samples) == len(local_audio)
    for sample, (name, source) in zip(samples, local_audio.items(), strict=True):
        assert set(sample) == set(mapping.values())
        assert sample["path"] == name
        assert sample["begin"] == BEGIN_TIME
        assert sample["end"] == END_TIME
        _assert_audio(
            sample["waveform"],
            sample["rate"],
            source,
            mono_method,
            sample_rate,
            load_audio_segments,
        )
