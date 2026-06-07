import numpy as np
import matplotlib.pyplot as plt

from filters.windowing import windows

N_fft = 2048

for name, w in windows.items():

    W = np.fft.fft(w, N_fft)

    W_db = 20*np.log10(
        np.abs(np.fft.fftshift(W)) /
        np.max(np.abs(W))
    )

    plt.plot(W_db, label=name)

plt.title("Window Frequency Responses")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude (dB)")
plt.ylim([-120,5])

plt.legend()
plt.grid(True)
plt.show()