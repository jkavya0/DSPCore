import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# MANUAL CONVOLUTION
# -----------------------
def manual_convolution(x, h):
    N = len(x)
    M = len(h)

    y = np.zeros(N + M - 1)

    for n in range(N + M - 1):
        for k in range(N):
            if (n - k) >= 0 and (n - k) < M:
                y[n] += x[k] * h[n - k]

    return y


# -----------------------
# PLOTTING
# -----------------------
def plot_signal(x, title):
    n = np.arange(len(x))
    plt.stem(n, x)
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    # Input signal
    x = np.array([1, 2, 3])

    # System impulse response
    h = np.array([1, 1, 1])

    # Manual convolution
    y_manual = manual_convolution(x, h)

    # Built-in convolution
    y_numpy = np.convolve(x, h)

    print("Manual:", y_manual)
    print("NumPy :", y_numpy)

    # Plot
    plot_signal(x, "Input Signal x[n]")
    plot_signal(h, "Impulse Response h[n]")
    plot_signal(y_manual, "Output y[n] = x[n] * h[n]")