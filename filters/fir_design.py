import numpy as np
import matplotlib.pyplot as plt

N = 51
fc = 0.2

n = np.arange(N)
h = 2 * fc * np.sinc(2 * fc * (n - (N-1)/2))

window = np.hamming(N)
h = h * window

plt.stem(h)
plt.title("FIR Low Pass Impulse Response")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.show()

H = np.fft.fft(h, 2048)

freq = np.linspace(
    0,
    0.5,
    len(H)//2)

plt.plot(
    freq,
    20*np.log10(
        np.abs(H[:len(H)//2])))

plt.title("FIR Low Pass Frequency Response")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.show()