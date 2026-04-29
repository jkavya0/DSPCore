import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# DTFT (numerical)
# -----------------------
def dtft(x, n, w):
    """
    x : signal values
    n : time index array
    w : frequency array
    """
    X = np.zeros(len(w), dtype=complex)

    for i in range(len(w)):
        for k in range(len(n)):
            X[i] += x[k] * np.exp(-1j * w[i] * n[k])
    return X

def plot_dtft(w, X, title):
    plt.figure(figsize=(10,4))

    # Magnitude
    plt.subplot(1,2,1)
    plt.plot(w, np.abs(X))
    plt.title("Magnitude")
    plt.xlabel("Frequency (rad/sample)")
    plt.grid()

    # Phase
    plt.subplot(1,2,2)
    plt.plot(w, np.angle(X))
    plt.title("Phase")
    plt.xlabel("Frequency (rad/sample)")
    plt.grid()

    plt.suptitle(title)
    plt.show()

if __name__ == "__main__":

    # Define signal
    n = np.arange(0, 5)
    x = np.array([1, 1, 1, 1, 1])   # rectangular sequence

    # Frequency range (-pi to pi)
    w = np.linspace(-np.pi, np.pi, 500)

    # Compute DTFT
    X = dtft(x, n, w)

    # Plot
    plot_dtft(w, X, "DTFT of Rectangular Signal")