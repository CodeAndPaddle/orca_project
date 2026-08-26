import librosa
import numpy as np

from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d, median_filter

#freqs between 5kHz and 20kHz
def bandpass_filter(y, sr, lowcut=5000, highcut=20000, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, y)


def compute_spectrogram(y, sr, n_fft=2048, hop_length=256):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    return S, freqs, times

# finding click frames as those with unusually broad spectrum and high energy.
def detect_click_frames(S_db):
    spread = np.percentile(S_db, 90, axis=0) - np.percentile(S_db, 10, axis=0)
    median_energy = np.median(S_db, axis=0)
    spread_thr = np.median(spread) + 1.5 * np.std(spread)
    energy_thr = np.median(median_energy) + 1.5 * np.std(median_energy)
    return (spread > spread_thr) & (median_energy > energy_thr)

# preparing spectogram for fidning contour
def preprocess_spectrogram(S, freqs, min_freq=5000, max_freq=20000):
    
    # taking 5kHz to 20kHz part of spectrogram
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    S_sub = S[freq_mask, :]
    freqs_sub = freqs[freq_mask]

    # converting to dB
    S_db = librosa.amplitude_to_db(S_sub, ref=np.max)

    # detecting clicks
    clicks = detect_click_frames(S_db)
    click_idx = np.where(clicks)[0]

    # if there are some clicks but not more than 50% of frames, remove them and interpolate
    if 0 < len(click_idx) < S_db.shape[1] * 0.5:
        clean_idx = np.where(~clicks)[0]
        if len(clean_idx) > 1:
            for freq_bin in range(S_db.shape[0]):
                S_db[freq_bin, click_idx] = np.interp(click_idx, clean_idx, S_db[freq_bin, clean_idx])

    # removing noise floor
    noise_floor = median_filter(S_db, size=(1, 51))
    S_clean = S_db - noise_floor

    # setting negative values to 0
    S_clean = np.maximum(S_clean, 0.0)

    # smoothing
    S_clean = gaussian_filter1d(S_clean, sigma=1.0, axis=0)

    return S_clean, freqs_sub, clicks

# finding spectral peaks in each frame
# NOTE: min_height and min_prominence are tuned for my dataset, might need adjustment for others
def extract_peaks_per_frame(S_clean, freqs_sub, click_mask, min_height=6.0, min_prominence=4.0, max_peaks=8):
    
    peaks_per_frame = []
    for j in range(S_clean.shape[1]):
        # skipping click frames
        if click_mask[j]:
            peaks_per_frame.append((np.array([]), np.array([])))
            continue

        # one column of spectrum
        col = S_clean[:, j]
        pks, props = find_peaks(col, height=min_height, prominence=min_prominence, distance=5)

        # if no peaks found
        if len(pks) == 0:
            peaks_per_frame.append((np.array([]), np.array([])))
            continue

        # if more than max_peaks found, take only the strongest ones
        if len(pks) > max_peaks:
            top_idx = np.argsort(props["peak_heights"])[-max_peaks:]
            pks = pks[top_idx]
            heights = props["peak_heights"][top_idx]
        # else take all peaks
        else:
            heights = props["peak_heights"]

        peaks_per_frame.append((freqs_sub[pks], heights))

    return peaks_per_frame
