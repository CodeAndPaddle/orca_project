<h1 align="center">orca_project</h1>

This project estimates the delay between the direct and reflected arrivals of
a synthetic dolphin-like whistle. The complete runtime pipeline is Python:

`synthetic WAV → time-frequency contour → analytic template → matched filter → delay`

MATLAB is retained only as the reference from which the contour algorithm was
ported. Running the notebook does not require MATLAB or a `.mat` file.

## Setup

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab Finding_Slant_delay.ipynb
```

Development and MATLAB-golden comparison dependencies are separate:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## Notebook workflow

`Finding_Slant_delay.ipynb` performs the following steps in order:

1. Generate a one-second, 48 kHz whistle whose instantaneous frequency follows
   the normalized contour `tan(x) - sin(x) + 1`, mapped from 4 to 10 kHz.
2. Save the source as `synthetic_dolphin_chirp.wav`.
3. Call the native Python extractor with the manually selected interval
   `[(0.0, SIGNAL_DURATION_S)]`.
4. Fit the extracted ridge, integrate it into an analytic template, and match
   the template against the noisy two-path received signal.
5. Measure the separation between the direct and reflected response peaks.

Run the notebook from top to bottom. The extractor cell must run after the WAV
writer cell because the WAV path is its input.

## Python contour API

```python
from birdcall_contour_bundle import (
    birdcall_contour_default_config,
    extract_birdcall_contours,
)

cfg = birdcall_contour_default_config()
cfg.output.dir = "outputs/example"
cfg.output.prefix = "example"

results = extract_birdcall_contours(
    "recording.wav",
    [(0.25, 1.10), (2.00, 2.75)],
    cfg,
)

if not results.contours:
    raise RuntimeError("No interval produced an accepted contour")

time_seconds = results.contours[0].time_sec_abs
frequency_hz = results.contours[0].freq_hz
```

Intervals are absolute seconds from the start of the WAV. They must be supplied
as a nonempty `N × 2` sequence. Bounds are clipped to the recording, while an
interval with no positive overlap is rejected. Version 1 intentionally does not
implement the MATLAB automatic-interval detector.

The default configuration includes audio-channel selection, bandpass, STFT,
robust enhancement, Viterbi tracking, confidence, output, and figure settings.
It is a nested dataclass, so values can be changed with assignments such as
`cfg.stft.win_samples = 512` or `cfg.audio.channel_mode = "first"`.

`Contour.path_scores` are signed robust-enhancement scores. Negative values are
valid and do not mean that a contour point is invalid. The notebook clips these
scores only when converting them to nonnegative least-squares weights.

## Output artifacts

For `recording.wav`, the default output directory is
`recording_birdcall_contours/` next to the WAV:

```text
recording_birdcall_contours/
├── figures/
│   └── recording_interval0001.png
├── recording_contour_summary.csv
├── recording_contour_points.csv
├── recording_interval_status.csv
├── recording_contours.npz
└── recording_run_metadata.json
```

The CSV files contain human-readable summaries, individual time-frequency
points, and per-interval acceptance/error status. The JSON file records the run
configuration and audio metadata.

The compressed NPZ is the stable machine-readable interchange format. It never
uses Python pickles. Multiple variable-length contours are represented by a
flattened point array and `contour_offsets`; the archive also stores interval
bounds/statuses, contour confidence, sample rate, absolute times, smoothed/raw
frequencies, and signed path scores. Use `load_contours_npz()` to validate and
load it.

## MATLAB reference

`birdcall_contour_bundle/extract_birdcall_contours.m` and
`birdcall_contour_default_config.m` remain available for reference and golden
comparison. The MATLAB configuration function name matches its filename. MATLAB
automatic detection remains reference-only and is outside the Python version-1
scope.

## Delay estimator notes

- The source, contour metadata, template, impulse response, and received signal
  must use the same sample rate.
- The matched-filter kernel is the conjugate-reversed, energy-normalized analytic
  template.
- Peak separation is measured relative to the detected direct arrival. The
  deterministic synthetic acceptance criterion is an error of at most two
  samples from the configured 20 ms delay.
