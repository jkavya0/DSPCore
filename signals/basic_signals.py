import numpy as np
import matplotlib.pyplot as plt

def unit_impulse(n, n0=0):
    x = np.zeros_like(n)
    x[n == n0] = 1
    return x

def unit_step(n):
    return (n >= 0).astype(int)

def ramp(n):
    return n * (n >= 0)

def triangular(n):
    return np.maximum(1 - np.abs(n) / max(abs(n)), 0)

def sinusoid(n, freq, fs):
    return np.sin(2 * np.pi * freq * n / fs)

def cosine(n, freq, fs):
    return np.cos(2 * np.pi * freq * n / fs)

def rectangular(n, width):
    return np.where(np.abs(n) <= width // 2, 1, 0)

def plot_signal(n, x, title):
    plt.plot(n, x)
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    n = np.arange(-20, 21)

    # Unit Impulse
    x1 = unit_impulse(n)
    plot_signal(n, x1, "Unit Impulse")

    # Unit Step
    x2 = unit_step(n)
    plot_signal(n, x2, "Unit Step")

    # Ramp
    x3 = ramp(n)
    plot_signal(n, x3, "Ramp Signal")

    # Triangular
    x4 = triangular(n)
    plot_signal(n, x4, "Triangular Signal")

    # Sinusoid
    x5 = sinusoid(n, freq=2, fs=20)
    plot_signal(n, x5, "Sinusoidal Signal")

    # Cosine
    x6 = cosine(n, freq=2, fs=20)
    plot_signal(n, x6, "Cosine Signal")

    # Rectangular
    x7 = rectangular(n, width=10)
    plot_signal(n, x7, "Rectangular Signal")