# Sampling Theorem (Nyquist-Shannon Theorem): A continuous-time signal can be perfectly reconstructed from its samples,
# if the sampling frequency is at least twice the highest frequency present in the signal.
# fs ≥ 2fmax

import numpy as np
import matplotlib.pyplot as plt

f_signal = 5

t = np.linspace(0, 1, 2000)

xc = np.sin(2 * np.pi * f_signal * t)

fs = 20

T = 1 / fs

k = np.arange(0, fs)

ts = k * T

# Sampled signal
xk = np.sin(2 * np.pi * f_signal * ts)

plt.figure(figsize=(10,5))

# Continuous signal
plt.plot(t, xc, label='Continuous Signal')

# Sampled signal
plt.stem(
    ts,
    xk,
    linefmt='r-',
    markerfmt='ro',
    basefmt='k-',
    label='Sampled Signal'
)

plt.title('Sampling Theorem Demonstration')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.grid(True)
plt.legend()

plt.show()