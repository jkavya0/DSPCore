# Sampling is the process of converting a continuous-time signal into a discrete-time sequence by taking signal values at uniformly spaced intervals T.
# The sampled signal is represented using an impulse train: xs(t)=−∞∑∞xc(kT).δ(t−kT)
# Suppose the continuous signal is xc(t)=sin(2πft)
# We are sample it every T seconds.

import numpy as np
import matplotlib.pyplot as plt

# signal frequency (Hz)
f = 2

#continuous-time axis
t = np.linspace(0,2,1000)

# Continuous signal
xc = np.sin(2 * np.pi * f * t)

# Sampling parameters
fs = 10
T = 1/fs

# Sample instants
k = np.arange(0,20)
ts = k * T

# Sampled sequence
xk = np.sin(2 * np.pi * f * ts)

# Plot continuous signal
plt.plot(t,xc,label = "Continuous signal")

# Plot sampled points
markerline, stemlines, baseline = plt.stem(ts, xk)

plt.setp(stemlines, color='red')
plt.setp(markerline, color='red')

markerline.set_label('Sampled Signal')
markerline.set_label('Sampled Signal')

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sampling of continuous-Time signal')
plt.legend()
plt.grid(True)
plt.show()



