# Chebyshev Filters : Sharper cutoff than Butterworth, introduces ripples,faster attenuation
#Chebyshev Type I → ripples in passband
#Chebyshev Type II → ripples in stopband

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (cheby1,cheby2, filtfilt,freqz)

#create a signal
fs = 1000
t = np.linspace(0, 1, fs)

# Mixed signal
x = (np.sin(2*np.pi*5*t) +
     0.5*np.sin(2*np.pi*50*t) +
     0.3*np.sin(2*np.pi*120*t))

# Chebyshev type I-LPF
fc = 40
b_cheby, a_cheby = cheby1(
    N=4,
    rp=4, #ripple parameter
    Wn=fc/(fs/2),
    btype='low'
)
y_chebyI = filtfilt(b_cheby,a_cheby,x)

# Chebyshev type II-LPF
fc = 20
b_cheby2, a_cheby2 = cheby2(
    N=4,
    rs=40,#stopband attenuatio
    Wn=fc/(fs/2),
    btype='low'
)
y_chebyII = filtfilt(b_cheby2,a_cheby2,x)
plt.figure(figsize=(14,8))

# Original
plt.subplot(3,1,1)
plt.plot(t, x)
plt.title("Original Signal")
plt.grid()
# Chebyshev type I
plt.subplot(3,1,2)
plt.plot(t, y_chebyI)
plt.title("Chebyshev Type-I Low Pass Output")
plt.grid()
# Chebyshev type II
plt.subplot(3,1,3)
plt.plot(t, y_chebyII)
plt.title("Chebyshev Type-II Low Pass Output")
plt.grid()

plt.tight_layout()
plt.show()

#Magintude response
w, h = freqz(b_cheby, a_cheby, worN=8000)
freq = w * fs / (2*np.pi)
plt.figure(figsize=(12,6))
plt.plot(
    freq,
    20*np.log10(np.abs(h) + 1e-10)
)
plt.title("Chebyshev Type-I Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, 200)
plt.ylim(-80, 5)
plt.grid()
plt.show()

#Magnitude response
w, h = freqz(b_cheby2, a_cheby2, worN=8000)
freq = w * fs / (2*np.pi)
plt.figure(figsize=(12,6))
plt.plot(
    freq,
    20*np.log10(abs(h))
)

plt.title("Chebyshev type II Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, 200)
plt.ylim(-80, 5)
plt.grid()
plt.show()

#Phase response
phase = np.angle(h)

plt.figure(figsize=(12,6))

plt.plot(freq, phase)

plt.title("Chebyshev Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

plt.xlim(0, 200)
plt.grid()
plt.show()