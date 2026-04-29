import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# SIGNAL (time-varying)
# -----------------------
def generate_signal(fs, duration):
    t = np.arange(0, duration, 1/fs)

    # frequency changes over time
    x = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*15*t*(t > duration/2))

    return t, x


# -----------------------
# STFT
# -----------------------
def stft(x, fs, window_size, hop_size):
    N = len(x)

    window = np.hanning(window_size)

    frames = []

    for start in range(0, N - window_size, hop_size):
        segment = x[start:start + window_size]

        # apply window
        segment = segment * window

        # FFT
        X = np.fft.fft(segment)

        frames.append(np.abs(X))

    return np.array(frames)


# -----------------------
# PLOT SPECTROGRAM
# -----------------------
def plot_spectrogram(S, fs, window_size, hop_size):
    plt.figure(figsize=(8,6))

    # Only positive frequencies
    S = S[:, :window_size//2]

    plt.imshow(S.T, aspect='auto', origin='lower',
               extent=[0, S.shape[0], 0, fs/2])

    plt.colorbar(label="Magnitude")
    plt.xlabel("Time Frame")
    plt.ylabel("Frequency (Hz)")
    plt.title("STFT Spectrogram")

    plt.show()


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    fs = 100
    duration = 2

    t, x = generate_signal(fs, duration)

    window_size = 64
    hop_size = 16

    S = stft(x, fs, window_size, hop_size)

    plot_spectrogram(S, fs, window_size, hop_size)