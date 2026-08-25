# Orca slant-delay experiment

This project generates a dolphin-like whistle, extracts its time-frequency
contour from a manually selected WAV interval, builds an analytic template, and
estimates a reflected-path delay after cancelling the stronger direct arrival.

```text
synthetic WAV → contour → analytic template → matched filter → delay
```

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab Finding_Slant_delay.ipynb
```

Run the notebook from top to bottom. Its single `ExperimentConfig` block is the
place to change the sample rate, duration, delay, noise, frequency band, search
bounds, and output paths.

```python
from pathlib import Path
from orca import ExperimentConfig

config = ExperimentConfig(
    sample_rate=24_001,
    duration_seconds=0.5,
    true_delay_seconds=0.02,
    noise_std=0.01,
    wav_path=Path("synthetic_dolphin_chirp.wav"),
    output_dir=Path("synthetic_dolphin_contours"),
    min_delay_seconds=0.002,
    max_delay_seconds=0.030,
)
```

Supported sample rates are 24,001–96,000 Hz and supported whistle durations
are 0.1–2.0 seconds. The 4–10 kHz default whistle and its analysis margin must
remain below Nyquist. The true delay is used only to generate and evaluate the
synthetic echo; the estimator searches the independent minimum and maximum
delay bounds.

## Adaptive STFT

`window_seconds` defaults to `256 / 48_000`, or 5.333 ms, because the validated
reference used a 256-sample window at 48 kHz. The extractor derives
`round(sample_rate * window_seconds)`, so it uses 256 samples at 48 kHz and 128
samples at 24,001 Hz. Do not divide 256 by the current sample rate: doing that
would always resolve back to 256 samples and would prevent the window duration
from adapting. The FFT size is the next power of two at least 16 times the
resolved window length.

## Extraction API and artifacts

```python
from orca import extract_contours

result = extract_contours(
    config.wav_path,
    intervals=[(0.0, config.duration_seconds)],
    output_dir=config.output_dir,
    config=config.contour_config(),
)
contour = next(item for item in result.contours if item.accepted)
```

Intervals are absolute seconds from the beginning of the WAV. Automatic
whistle detection is not included; the extractor tracks one contour across
each entire supplied interval.

Each run writes only:

```text
synthetic_dolphin_contours/
├── contours.npz
└── figures/
    └── interval_0001.png
```

`contours.npz` is pickle-free and stores the sample rate, input path,
configuration, intervals, acceptance/confidence values, contour offsets,
timestamps, frequencies, and signed enhancement path scores. Load it with
`numpy.load(path, allow_pickle=False)`.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```
