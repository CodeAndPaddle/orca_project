import numpy as np


def viterbi_peaks(peaks_per_frame, max_jump_hz=1200, gap_penalty=12.0, transition_weight=0.015):

    n_frames = len(peaks_per_frame)

    # finding first frame with at least one peak
    start_frame = None
    for j in range(n_frames):
        pf, pe = peaks_per_frame[j]
        if len(pf) > 0:
            start_frame = j
            break
    if start_frame is None:
        return np.full(n_frames, np.nan)

    # initializing Viterbi algorithm
    pf, pe = peaks_per_frame[start_frame]
    prev_states = {}
    for k in range(len(pf)):
        prev_states[k] = (gap_penalty - pe[k], pf[k])
    prev_states[-1] = (0.0, np.nan)

    # backpointers for path reconstruction
    backpointers = []
    for frame in range(n_frames):
        backpointers.append({})

    # running Viterbi algorithm
    for j in range(start_frame + 1, n_frames):
        # all peaks in current frame
        pf_cur, pe_cur = peaks_per_frame[j]
        cur_states = {}

        # iterating through all peaks in current frame and looking for best match in previous frame
        for ci in range(len(pf_cur)):
            f_cur = pf_cur[ci]
            e_cur = pe_cur[ci]
            best_cost = np.inf
            best_prev = -1

            # iterating through all peaks in previous frame and trying to match them
            for pi, (prev_cost, prev_freq) in prev_states.items():
                
                # previous state is silence
                if pi == -1 or np.isnan(prev_freq):
                    cost = prev_cost + gap_penalty - e_cur
                
                else:
                    # previous state is a peak
                    df = abs(f_cur - prev_freq)
                    if df > max_jump_hz:
                        continue
                    # penalty for jumping
                    trans = transition_weight * (df ** 2) / max_jump_hz
                    cost = prev_cost + trans - e_cur

                # if current peak is better than best peak
                if cost < best_cost:
                    best_cost = cost
                    best_prev = pi

            # if there is path to that peak save it
            if best_cost < np.inf:
                cur_states[ci] = (best_cost, f_cur)
                backpointers[j][ci] = (best_prev, j - 1)

        # silence state for current frame
        best_sil_cost = np.inf
        best_sil_prev = -1
        for pi, (prev_cost, prev_freq) in prev_states.items():
            cost = prev_cost if pi == -1 else prev_cost + gap_penalty
            if cost < best_sil_cost:
                best_sil_cost = cost
                best_sil_prev = pi

        cur_states[-1] = (best_sil_cost, np.nan)
        backpointers[j][-1] = (best_sil_prev, j - 1)
        prev_states = cur_states

    # at the and take the best final state
    best_idx = -1
    best_cost = np.inf
    for pi, (cost, freq) in prev_states.items():
        if cost < best_cost:
            best_cost = cost
            best_idx = pi

    # reconstructing the path
    contour = np.full(n_frames, np.nan)
    j = n_frames - 1
    ci = best_idx
    while j >= 0:
        # if current state is a peak and not silence, save frequency
        if ci != -1:
            pf_j, pe_j = peaks_per_frame[j]
            if 0 <= ci < len(pf_j):
                contour[j] = pf_j[ci]
        # move to previous state
        if j < len(backpointers) and ci in backpointers[j]:
            prev_ci, prev_j = backpointers[j][ci]
            ci = prev_ci
            j = prev_j
        else:
            break

    return contour
