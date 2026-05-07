# Quantization is the process of mapping a continuous range of amplitude values into a finite set of discrete numerical levels.

import numpy as np
import matplotlib.pyplot as plt

fs = 1000          # Sampling frequency
t = np.linspace(0, 1, fs)

# Original continuous signal
x = np.sin(2 * np.pi * 5 * t)

bits = 2                     # Number of bits
L = 2 ** bits                # Number of quantization levels

x_min = -1
x_max = 1

# Quantization step size
delta = (x_max - x_min) / L

# Quantization formula
xq = np.round(x / delta) * delta

# Quantization error
error = x - xq

plt.figure(figsize=(12,8))

# Original vs Quantized
plt.subplot(3,1,1)
plt.plot(t, x, label='Original Signal')
plt.step(t, xq, where='mid', label='Quantized Signal')
plt.title('Quantization')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Quantization Error
plt.subplot(3,1,2)
plt.plot(t, error)
plt.title('Quantization Error')
plt.xlabel('Time')
plt.ylabel('Error')
plt.grid()

# Staircase View
plt.subplot(3,1,3)
plt.step(t, xq, where='mid')
plt.title('Discrete Quantized Levels')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()