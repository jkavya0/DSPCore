import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# SIGNAL (non-bin-aligned)
# -----------------------
def generate_signal(fs, duration):
    t = np.arange(0, duration, 1/fs)

    f = 5 # not aligned with FFT bins → leakage
    x = np.sin(2 * np.pi * f * t)

    return t, x


# -----------------------
# WINDOWS
# -----------------------
def rectangular_window(N):
    return np.ones(N)


def hanning_window(N):
    return np.hanning(N)


def hamming_window(N):
    return np.hamming(N)


# -----------------------
# FFT
# -----------------------
def compute_fft(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    freq = np.fft.fftfreq(N, d=1/fs)

    return freq, X


# -----------------------
# PLOTTING
# -----------------------
def plot_spectrum(freq, X, title):
    N = len(X)
    half = N // 2

    magnitude = np.abs(X) / N

    plt.plot(freq[:half], magnitude[:half])
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    fs = 50
    duration = 1

    t, x = generate_signal(fs, duration)
    N = len(x)

    # Windows
    w_rect = rectangular_window(N)
    w_hann = hanning_window(N)
    w_hamm = hamming_window(N)

    # Apply windows
    x_rect = x * w_rect
    x_hann = x * w_hann
    x_hamm = x * w_hamm

    # FFT
    f1, X_rect = compute_fft(x_rect, fs)
    f2, X_hann = compute_fft(x_hann, fs)
    f3, X_hamm = compute_fft(x_hamm, fs)

    # Plot comparison
    plt.figure(figsize=(10,6))

    plt.subplot(3,1,1)
    plot_spectrum(f1, X_rect, "Rectangular Window (Leakage)")

    plt.subplot(3,1,2)
    plot_spectrum(f2, X_hann, "Hanning Window")

    plt.subplot(3,1,3)
    plot_spectrum(f3, X_hamm, "Hamming Window")

    plt.tight_layout()
    plt.show()