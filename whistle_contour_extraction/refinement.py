import numpy as np
from scipy.signal import medfilt, savgol_filter

# remove high frequency noise from contour
def roughness_filter(contour, window=15, max_roughness_hz=800):

    # taking contour length and making mask of all true values
    n = len(contour)
    mask = np.ones(n, dtype=bool)

    # iterating through each frame
    for i in range(n):
        # if frame is NaN, skip it
        if np.isnan(contour[i]):
            mask[i] = False
            continue

        # taking local window around current frame
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        segment = contour[lo:hi]

        # removing NaN values from segment
        valid_seg = segment[~np.isnan(segment)]

        # if less than 3 valid values, skip it
        if len(valid_seg) < 3:
            continue

        # calculating roughness
        diffs = np.abs(np.diff(valid_seg))
        roughness = np.std(diffs)

        # if roughness is too high, remove the frame
        if roughness > max_roughness_hz:
            mask[i] = False

    # applying mask
    result = contour.copy()
    result[~mask] = np.nan
    return result


def refine_contour(contour, times, min_duration=0.10, max_gap_frames=10):
    contour = contour.copy()

    contour = roughness_filter(contour, window=15, max_roughness_hz=600)

    #getting indices of all valid values
    valid_idx = np.where(~np.isnan(contour))[0]
    if len(valid_idx) < 10:
        return None

    # splitting contour into groups
    split_points = np.where(np.diff(valid_idx) > (max_gap_frames + 1))[0] + 1
    groups = np.split(valid_idx, split_points)

    best_group = None
    best_score = -np.inf

    # finding the best group
    for g in groups:

        if len(g) < 10:
            continue
        
        # calculating duration
        duration = times[g[-1]] - times[g[0]]
        if duration < min_duration:
            continue
        
        # calculating density
        seg = contour[g]
        span = g[-1] - g[0] + 1
        density = len(g) / span

        # calculating score
        score = len(g) * density + 3.0 * duration

        # penalizing roughness
        if len(seg) > 1:
            roughness = np.nanstd(np.diff(seg))
            score -= 0.002 * roughness

        if score > best_score:
            best_score = score
            best_group = g

    if best_group is None:
        return None

    # taking the best group (part of contour)
    start = best_group[0]
    end = best_group[-1]

    segment = contour[start:end + 1]
    seg_valid = ~np.isnan(segment)

    if np.sum(seg_valid) < 10:
        return None

    # interpolating and smoothing the segment
    x = np.arange(len(segment))
    segment_interp = np.interp(x, x[seg_valid], segment[seg_valid])
    segment_med = medfilt(segment_interp, kernel_size=5)

    # Savitzky-Golay smoothing
    win = min(21, len(segment_med))
    if win % 2 == 0:
        win -= 1
    if win >= 5:
        segment_smooth = savgol_filter(segment_med, win, 3)
    else:
        segment_smooth = segment_med

    contour_clean = np.full(len(contour), np.nan, dtype=float)
    contour_clean[start:end + 1] = segment_smooth

    return contour_clean
