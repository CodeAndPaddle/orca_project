import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def visualize_contour(S, freqs, times, sr, contour_clean, hop_length=256, title="Spectrogram + Contour", save_path=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap="magma")

    if contour_clean is not None:
        valid = ~np.isnan(contour_clean)
        ax.plot(times[valid], contour_clean[valid], color="cyan", linewidth=2.5, alpha=0.9)

    ax.set_ylim(0, 25000)
    ax.set_title(title, fontsize=13)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

    return fig
