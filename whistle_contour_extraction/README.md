# Dolphin Whistle Contour Extraction

This folder is now a compatibility/example entry point. The maintained
implementation lives in `Slant_delay_utills`; `main.py` delegates to its
automatic multi-whistle detector so the repository has one production
algorithm.

Automatic extraction of dolphin whistle frequency contours from underwater audio recordings using Viterbi-based spectral peak tracking.

## Folder Structure

```
whistle_contour/
├── main.py
├── preprocessing.py
├── tracking.py
├── refinement.py
├── visualization.py
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   └── dolphin_whistle_waw_dataset/
│       ├── 000204_2145_1.15.wav
│       ├── 191120-001_0.58.wav
│       └── ...
└── output/
    ├── 000204_2145_1.15.png
    └── ...
```

Place your `.wav` files in `data/dolphin_whistle_waw_dataset/` (or any subdirectory under `data/`).

The `output/` folder is created automatically and contains spectrogram PNG images with the extracted contour.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Single file:**
```bash
python main.py data/dolphin_whistle_waw_dataset/000204_2145_1.15.wav
```

The compatibility script prints every detected interval and confidence. Use
`Finding_Slant_delay.ipynb` for contour and matched-filter plots and structured
delay output.

## Project Structure

| File | Description |
|---|---|
| `main.py` | Entry point — contour extraction pipeline and CLI |
| `preprocessing.py` | Bandpass filter, spectrogram, click removal, noise subtraction, peak extraction |
| `tracking.py` | Viterbi algorithm for optimal path through spectral peaks |
| `refinement.py` | Roughness filter, segment selection, smoothing |
| `visualization.py` | Spectrogram + contour overlay plotting |


## Pipeline

1. **Bandpass filter** (5–20 kHz) — isolate whistle frequency range
2. **STFT spectrogram** — time-frequency representation
3. **Click detection** — identify and interpolate over echolocation clicks
4. **Noise subtraction** — median filter removes background noise
5. **Peak extraction** — find strongest spectral peaks per frame
6. **Viterbi tracking** — dynamic programming to find optimal frequency path
7. **Roughness filter** — remove jagged noise-tracking segments
8. **Segment selection + smoothing** — keep best contiguous segment, apply Savitzky-Golay filter

## Viterbi Algorithm

The [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) is a dynamic programming method originally designed for decoding signals in hidden Markov models. It finds the **most likely sequence of hidden states** given a series of observations, by efficiently searching through all possible state paths without brute-force enumeration.

### How we use it here

In this project, each time frame of the spectrogram is a "step" and the candidate spectral peaks in that frame are the possible "states". The Viterbi algorithm finds the optimal path through peaks across all frames by minimizing a total cost function.

**Three cost components determine the path:**

| Component | What it does | Effect |
|---|---|---|
| **Energy reward** | Subtract peak height from cost | Stronger peaks are preferred |
| **Transition penalty** | Cost proportional to frequency jump squared | Smooth, gradual frequency changes preferred over large jumps |
| **Gap penalty** | Fixed cost to enter/exit active tracking | Prevents the path from activating on isolated noise peaks |

### Adaptations for dolphin whistles

Standard Viterbi assumes one state per frame. My version adds:

- **Silence state** — the algorithm can choose to "not track" in a given frame. Staying silent is free, but entering/leaving silence costs `gap_penalty = 12`. This means the algorithm only starts tracking when there is a sustained series of strong peaks (a real whistle), not isolated noise.
- **Sparse peak candidates** — instead of considering every frequency bin (~350 bins), we only consider the top 8 spectral peaks per frame. This dramatically reduces computation and avoids noise.
- **Max frequency jump** — transitions larger than 1200 Hz between consecutive frames are forbidden, preventing the path from jumping across unrelated signals.
- **Post-hoc roughness filter** — after the Viterbi path is found, we measure local frequency-jump roughness in a sliding window. Real whistles evolve smoothly (low roughness), while noise-tracking segments jump wildly (high roughness > 600 Hz). High-roughness frames are removed.
