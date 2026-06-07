import numpy as np
import matplotlib.pyplot as plt

N = 64

windows = {
    "Rectangular": np.ones(N),
    "Hamming": np.hamming(N),
    "Hann": np.hanning(N),
    "Blackman": np.blackman(N)
}

plt.figure(figsize=(10,6))

for name, w in windows.items():
    plt.plot(w, label=name)

plt.title("Window Functions")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.show()