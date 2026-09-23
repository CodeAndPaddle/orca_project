# Orca slant-delay experiment

## Unknown-time WAV workflow

The primary workflow now accepts a complete WAV without manually supplied
whistle intervals. It detects every separated whistle, extracts a contour,
builds a contour-guided phase template, and searches for credible reflected
paths after cancelling the dominant direct arrival.

```python
from pathlib import Path

from Slant_delay_utills import (
    AnalysisConfig,
    analyze_detections,
    analyze_recording,
    detect_whistles,
)

config = AnalysisConfig(
    whistle_band_hz=(3_500.0, 10_500.0),  # required; must be below Nyquist
    min_delay_seconds=0.002,
    max_delay_seconds=0.030,
    max_echoes=4,
    output_dir=Path("whistle_analysis"),
)
analysis = analyze_recording("unknown_recording.wav", config)

for whistle in analysis.whistles:
    print(whistle.detection.start_seconds, whistle.detection.end_seconds)
    print(whistle.delay_seconds, whistle.delay_confidence, whistle.warnings)
```

For a visible staged workflow, detect once and pass those immutable results to
the analysis stage. This is the flow demonstrated in the notebook:

```python
detections = detect_whistles("unknown_recording.wav", config)
analysis = analyze_detections("unknown_recording.wav", config, detections)
```

For a reproducible notebook test, run `Synthetic_Slant_Delay_Validation.ipynb`
first. It generates `synthetic_dolphin_chirp.wav` with 10 ms and 20 ms echoes,
saves its ground truth separately, and verifies the estimator. Then run
`Finding_Slant_delay.ipynb`, which independently detects and analyzes that WAV
without using the ground truth during estimation.

Each analysis run stores a pickle-free `analysis.npz`. The detector supports
multiple separated whistles; simultaneous overlapping whistles remain an
ambiguous single-path case. Real-recording delay estimates are experimental
until validated against labeled data.

This project generates a dolphin-like whistle, extracts its time-frequency
contour from a manually selected WAV interval, builds an analytic template, and
estimates multiple reflected-path delays after cancelling the stronger direct arrival.

```text
synthetic WAV → contour → analytic template → matched filter → delays
```
Goal

- Generate an artificial signal and multiple delayed copies of it.
- Estimate the slant delays $\tau_k$ from the resulting filter response.

Approach

1. Build a synthetic source signal $s(t)$, for example a short whistle like waveform.
2. Define a filter $h(t)$ that combines a direct path with multiple delayed copies.
   - Example: $h(t)=0.5\,\delta(t)+0.2\,\delta(t-0.01)+0.15\,\delta(t-0.02)$
3. Form the received signal $r(t)=s(t) * h(t)$.
4. Add a small amount of Gaussian noise to the received signal, e.g. $r(t) + 0.01\,\text{noise}$.
5. Estimate the time-frequency contour of the synthetic signal $\tilde S(t)$ using the Lixing code.
6. Correlate or convolve $\tilde S(t)$ with $r(t)$ to recover an approximate impulse response $\tilde h(t)$.
   - The goal is that $\tilde S(t) * r(t) \approx \delta(t) * h(t) = h(t)$.
7. Identify peaks in $\tilde h(t)$ and compute their time differences from the direct peak.
   - These time differences are the slant delays $\tau_k$.

Notes

- Use the same sampling rate for $s(t)$ and $h(t)$ so both are consistent as sampled signals.
- Keep the noise level low so the delay estimation remains reliable.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab Finding_Slant_delay.ipynb
```

The notebook uses `AnalysisConfig` for unknown-time WAV files. The
`ExperimentConfig` below remains available for controlled synthetic experiments
and regression tests.

```python
from pathlib import Path
from Slant_delay_utills import ExperimentConfig

config = ExperimentConfig(
    sample_rate=24_001,
    duration_seconds=0.5,
    true_delay_seconds=[0.01, 0.02],
    direct_gain=0.5,
    echo_gains=[0.2, 0.15],
    noise_std=0.01,
    wav_path=Path("synthetic_dolphin_chirp.wav"),
    output_dir=Path("synthetic_dolphin_contours"),
    min_delay_seconds=0.002,
    max_delay_seconds=0.030,
    min_path_separation_seconds=0.002,
)
```

Supported sample rates are 24,001–96,000 Hz and supported whistle durations
are 0.1–2.0 seconds. The 4–10 kHz default whistle and its analysis margin must
remain below Nyquist. The true delays are used only to generate and evaluate
the synthetic echoes; the estimator receives only the requested echo count and
searches the independent minimum and maximum delay bounds. Returned paths are
ordered by arrival time.

## Multi-path synthesis and estimation

```python
from Slant_delay_utills import estimate_delays, synthesize_multipath

received, impulse_response = synthesize_multipath(
    source,
    config.delay_samples,
    config.echo_gains,
    direct_gain=config.direct_gain,
    noise_std=config.noise_std,
    seed=7,
)
estimate = estimate_delays(
    received,
    template,
    config.sample_rate,
    num_echoes=len(config.true_delay_seconds),
    min_delay_seconds=config.min_delay_seconds,
    max_delay_seconds=config.max_delay_seconds,
    min_path_separation_seconds=config.min_path_separation_seconds,
)
```

`estimate.delay_samples`, `estimate.delay_seconds`, and
`estimate.echo_indices` are one-dimensional arrays aligned by path. If fewer
credible peaks are found than requested, `estimate_delays` raises a
`RuntimeError` instead of returning a partial result.

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
from Slant_delay_utills import extract_contours

result = extract_contours(
    config.wav_path,
    intervals=[(0.0, config.duration_seconds)],
    output_dir=config.output_dir,
    config=config.contour_config(),
)
contour = next(item for item in result.contours if item.accepted)
```

Intervals are absolute seconds from the beginning of the WAV. This lower-level
API remains manual; use `detect_whistles` or `analyze_recording` when interval
locations are unknown.

For delay work, pass bandpassed reference samples to `template_from_contour` so
the contour defines the trajectory and interval while the analytic phase is
recovered from the recording. `analyze_recording` performs this automatically.

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
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```
