import os
import numpy as np
import librosa

from preprocessing import (
    bandpass_filter,
    compute_spectrogram,
    preprocess_spectrogram,
    extract_peaks_per_frame,
)
from tracking import viterbi_peaks
from refinement import refine_contour
from visualization import visualize_contour


def extract_contour(path_to_wav, min_freq=5000, max_freq=20000,n_fft=2048, hop_length=256):
    
    # audio loading
    y, sampling_rate = librosa.load(path_to_wav, sr=None)

    # bandpass filtering of signal
    y_filt = bandpass_filter(y, sampling_rate, lowcut=min_freq, highcut=max_freq)

    # spectrogram computation
    S, freqs, times = compute_spectrogram(y_filt, sampling_rate, n_fft=n_fft, hop_length=hop_length)

    # spectrogram preprocessing
    S_clean, freqs_sub, click_mask = preprocess_spectrogram(S, freqs, min_freq, max_freq)

    # peak extraction from every frame
    peaks_per_frame = extract_peaks_per_frame(S_clean, freqs_sub, click_mask, min_height=6.0, min_prominence=4.0, max_peaks=8)

    # viterbi tracking
    contour = viterbi_peaks(peaks_per_frame, max_jump_hz=1200, gap_penalty=12.0, transition_weight=0.015)

    # contour refinement
    contour_clean = refine_contour(contour, times, min_duration=0.10, max_gap_frames=10)

    return contour_clean, times, S, freqs, sampling_rate


def process_single(path_to_wav, save_path=None):

    # extract contour
    contour_clean, times, S, freqs, sampling_rate = extract_contour(path_to_wav)

    # just getting filename
    fname = os.path.basename(path_to_wav)

    # if contour is not None, get some stats
    if contour_clean is not None:
        # taking only valid points (not NaN or inf)
        valid_mask = np.isfinite(contour_clean)
        valid_freqs = contour_clean[valid_mask]
        valid_times = times[valid_mask]

        #stats
        num_points = len(valid_freqs)
        duration = valid_times[-1] - valid_times[0]
        freq_min = np.min(valid_freqs)
        freq_max = np.max(valid_freqs)

        status = f"OK  {num_points} pts, {duration:.2f}s, {freq_min:.0f}-{freq_max:.0f} Hz"
    else:
        status = "NO WHISTLE DETECTED"

    # visualizing contour on spectrogram
    title = f"{fname}  [{status}]"
    visualize_contour(S, freqs, times, sampling_rate, contour_clean, title=title, save_path=save_path)
    print(title)

    return contour_clean

# same thing just for the whole folder of waws (batch)
def process_batch(data_dir):

    # creating output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # getting all wav files
    wav_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    wav_files = sorted(wav_files)

    print(f"Found {len(wav_files)} WAV files in {data_dir}")
    print(f"Saving visualizations to {output_dir}/")
    print("-----------------------------------------------")

    detected = 0
    failed = 0

    for i, path in enumerate(wav_files):
        fname = os.path.basename(path)
        save_path = os.path.join(output_dir, fname.replace(".wav", ".png"))

        try:
            contour_clean, times, S, freqs, sampling_rate = extract_contour(path)

            if contour_clean is not None:
                valid_mask = np.isfinite(contour_clean)
                valid_freqs = contour_clean[valid_mask]
                valid_times = times[valid_mask]

                num_points = len(valid_freqs)
                duration = valid_times[-1] - valid_times[0]
                freq_min = np.min(valid_freqs)
                freq_max = np.max(valid_freqs)

                status = f"OK  {num_points:4d} pts  {duration:.2f}s  {freq_min:.0f}-{freq_max:.0f} Hz"
                detected += 1
            else:
                status = "NO WHISTLE"
                failed += 1

            title = f"{fname}  [{status}]"
            visualize_contour(S, freqs, times, sampling_rate, contour_clean, title=title, save_path=save_path)

            print(f"[{i+1}/{len(wav_files)}] {fname} {status}")

        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(wav_files)}] {fname} ERROR: {e}")

    print()
    print("-----------------------------------------------")
    total = len(wav_files)
    print(f"Detected: {detected}/{total} ({100 * (detected/total):.1f}%)")
    print(f"Failed:   {failed}/{total} ({100 * (failed/total):.1f}%)")


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    
    # main.py path/to/file.wav
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        wav_path = sys.argv[1]
        save_path = wav_path.replace(".wav", ".png")
        process_single(wav_path, save_path=save_path)

    # main.py path/to/folder
    else:
        process_batch(sys.argv[1])
