import numpy as np
import matplotlib.pyplot as plt

# Continuous signal
def continuous_signal(t, freq):
    return np.sin(2 * np.pi * freq * t)

# Sampled signal
def sampled_signal(freq, fs):
    t = np.arange(0, 1, 1/fs)
    x = np.sin(2 * np.pi * freq * t)
    return t, x

def plot_sampled(t, x, title):
    plt.stem(t, x)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# MAIN
if __name__ == "__main__":
    freq = 5

    # See ONLY sampled signals
    t1, x1 = sampled_signal(freq, 20)
    plot_sampled(t1, x1, "fs = 20 Hz (Good)")

    t2, x2 = sampled_signal(freq, 10)
    plot_sampled(t2, x2, "fs = 10 Hz (Critical)")

    t3, x3 = sampled_signal(freq, 6)
    plot_sampled(t3, x3, "fs = 6 Hz (Aliasing)")

