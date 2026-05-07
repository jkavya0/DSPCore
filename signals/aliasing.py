# Aliasing occurs when:
# fs <2fmax
# The sampling rate is too low.
# High-frequency components appear as lower frequencies.

import numpy as np
import matplotlib.pyplot as plt

f_signal = 8

t = np.linspace(0, 1, 2000)

xc = np.sin(2 * np.pi * f_signal * t)

fs = 10                              # Too low

T = 1 / fs

k = np.arange(0, fs)

ts = k * T

# Sampled sequence
xk = np.sin(2 * np.pi * f_signal * ts)

plt.figure(figsize=(10,5))

plt.plot(t, xc, label='Continuous Signal')

plt.stem(
    ts,
    xk,
    linefmt='m-',
    markerfmt='mo',
    basefmt='k-',
    label='Aliased Samples'
)

plt.title('Aliasing Demonstration')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.grid(True)
plt.legend()

plt.show()