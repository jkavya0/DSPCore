import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# GENERATE SIGNAL
# -----------------------
def generate_signal(fs, duration):
    t = np.arange(0, duration, 1/fs)

    # single tone (you can add more later)
    f1 = 5  # Hz
    x = np.sin(2 * np.pi * f1 * t)

    return t, x


# -----------------------
# FFT COMPUTATION
# -----------------------
def compute_fft(x, fs):
    N = len(x)

    X = np.fft.fft(x)
    freq = np.arange(N) * fs / N

    return freq, X


# -----------------------
# PLOTTING
# -----------------------
def plot_fft(freq, X, fs, title):
    N = len(X)

    # Only positive frequencies (real signal symmetry)
    half = N // 2

    plt.figure(figsize=(10,4))

    plt.plot(freq[:half], np.abs(X[:half]))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()

    plt.show()


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    fs = 50        # sampling frequency
    duration = 1   # seconds

    t, x = generate_signal(fs, duration)

    freq, X = compute_fft(x, fs)

    plot_fft(freq, X, fs, "FFT Spectrum")