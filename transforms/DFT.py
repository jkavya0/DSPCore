import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# MANUAL DFT
# -----------------------
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)

    return X


# -----------------------
# PLOTTING
# -----------------------
def plot_dft(X, title):
    N = len(X)
    k = np.arange(N)

    # Magnitude
    magnitude = np.abs(X)

    # Phase (fix: remove noise where magnitude ~ 0)
    phase = np.angle(X)
    phase[magnitude < 1e-6] = 0   # important fix

    plt.figure(figsize=(10,4))

    # Magnitude plot
    plt.subplot(1,2,1)
    plt.stem(k, magnitude)
    plt.title("Magnitude")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.grid()

    # Phase plot
    plt.subplot(1,2,2)
    plt.stem(k, phase)
    plt.title("Phase")
    plt.xlabel("k")
    plt.ylabel("Phase (radians)")
    plt.grid()

    plt.suptitle(title)
    plt.show()


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":

    # Example signal
    x = np.array([1, -1, 1, -1])

    # Compute DFT
    X_manual = dft(x)

    # Compare with NumPy FFT
    X_fft = np.fft.fft(x)

    print("Manual DFT:", X_manual)
    print("NumPy FFT :", X_fft)

    # Plot
    plot_dft(X_manual, "DFT of Rectangular Signal")