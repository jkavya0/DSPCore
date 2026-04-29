import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# SIGNAL GENERATION
# -----------------------
def generate_signal(fs, duration, freqs, noise_level=0):
    t = np.arange(0, duration, 1/fs)
    x = np.zeros_like(t)

    for f in freqs:
        x += np.sin(2 * np.pi * f * t)

    if noise_level > 0:
        x += noise_level * np.random.randn(len(t))

    return t, x


# -----------------------
# FFT + FREQUENCY AXIS
# -----------------------
def compute_fft(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    freq = np.fft.fftfreq(N, d=1/fs)

    return freq, X


# -----------------------
# FIND PEAK FREQUENCY
# -----------------------
def find_peak_frequency(x, fs):
    freq, X = compute_fft(x, fs)

    magnitude = np.abs(X)
    half = len(X) // 2

    peak_idx = np.argmax(magnitude[:half])
    return freq[peak_idx]


# -----------------------
# PLOT SPECTRUM
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
# FREQUENCY RESOLUTION
# -----------------------
def frequency_resolution(fs, N):
    return fs / N


# -----------------------
# ZERO PADDING
# -----------------------
def zero_pad(x, new_length):
    return np.pad(x, (0, new_length - len(x)))


# -----------------------
# MAIN DEMOS
# -----------------------
if __name__ == "__main__":

    fs = 50
    duration = 1

    # -----------------------
    # 1. SINGLE TONE
    # -----------------------
    t, x = generate_signal(fs, duration, [5])
    peak = find_peak_frequency(x, fs)
    print("Single tone peak:", peak)

    freq, X = compute_fft(x, fs)

    plt.figure()
    plot_spectrum(freq, X, "Single Tone (5 Hz)")
    plt.show()

    # -----------------------
    # 2. MULTI-TONE
    # -----------------------
    t, x = generate_signal(fs, duration, [5, 12])

    freq, X = compute_fft(x, fs)

    plt.figure()
    plot_spectrum(freq, X, "Multi-tone (5 Hz + 12 Hz)")
    plt.show()

    # -----------------------
    # 3. LEAKAGE DEMO
    # -----------------------
    t, x = generate_signal(fs, duration, [5.5])

    freq, X = compute_fft(x, fs)

    plt.figure()
    plot_spectrum(freq, X, "Leakage (5.5 Hz)")
    plt.show()

    # -----------------------
    # 4. WINDOW EFFECT
    # -----------------------
    window = np.hanning(len(x))
    x_windowed = x * window

    freq, Xw = compute_fft(x_windowed, fs)

    plt.figure()
    plot_spectrum(freq, X, "Without Window")
    plot_spectrum(freq, Xw, "With Hann Window")
    plt.legend(["Rectangular", "Hann"])
    plt.show()

    # -----------------------
    # 5. FREQUENCY RESOLUTION
    # -----------------------
    N = len(x)
    print("Frequency resolution:", frequency_resolution(fs, N))

    # -----------------------
    # 6. ZERO PADDING
    # -----------------------
    x_pad = zero_pad(x, 256)

    freq_pad, X_pad = compute_fft(x_pad, fs)

    plt.figure()
    plot_spectrum(freq, X, "Original")
    plot_spectrum(freq_pad, X_pad, "Zero-padded")
    plt.legend(["Original", "Zero-padded"])
    plt.show()

    # -----------------------
    # 7. NOISE ANALYSIS
    # -----------------------
    t, x = generate_signal(fs, duration, [5], noise_level=0.5)

    freq, X = compute_fft(x, fs)

    plt.figure()
    plot_spectrum(freq, X, "Signal + Noise")
    plt.show()