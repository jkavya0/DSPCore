# Butterworth filter : Maximally flat frequency response in passband.
# very smooth response , no ripples in passband.
# smooth, stable

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (butter,filtfilt,freqz)

#create a signal
fs = 1000
t = np.linspace(0, 1, fs)

# Mixed signal
x = (np.sin(2*np.pi*5*t) +
     0.5*np.sin(2*np.pi*50*t) +
     0.3*np.sin(2*np.pi*120*t))

#Butterworth LPF
fc = 20
b_butter, a_butter = butter(N=4,Wn=fc/(fs/2),btype='low')
y_low = filtfilt(b_butter,a_butter,x)
#Buterworth HPF
fc = 40
b_butter, a_butter = butter(N=4,Wn=fc/(fs/2),btype='high')
y_high = filtfilt(b_butter,a_butter,x)

#compute fft
def compute_fft(signal, fs):
    N = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(N, 1/fs)
    magnitude = np.abs(fft_vals)
    # Keep positive frequencies only
    half = N // 2
    return (
        fft_freq[:half],
        magnitude[:half]
    )
# comput spectrogram
f_x, mag_x = compute_fft(x, fs)
f_bl, mag_bl = compute_fft(y_low, fs)
f_bh, mag_bh = compute_fft(y_high, fs)

plt.figure(figsize=(14,8))

# Original
plt.subplot(3,1,1)
plt.plot(f_x, mag_x)
plt.title("FFT - Original Signal")
plt.xlim(0, 200)
plt.grid()
# Butterworth LPF
plt.subplot(3,1,2)
plt.plot(f_bl, mag_bl)
plt.title("FFT - Butterworth LPF")
plt.xlim(0, 200)
plt.grid()
# Butterworth HPF
plt.subplot(3,1,3)
plt.plot(f_bh, mag_bh)
plt.title("FFT - Butterworth HPF")
plt.grid()

plt.tight_layout()
plt.show()

#Magnitude response
w, h = freqz(b_butter, a_butter, worN=8000)

freq = w * fs / (2*np.pi)

plt.figure(figsize=(12,6))

plt.plot(
    freq,
    20*np.log10(abs(h))
)

plt.title("Butterworth Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(0, 200)
plt.grid()

plt.show()

#Phase response
phase = np.angle(h)

plt.figure(figsize=(12,6))

plt.plot(freq, phase)

plt.title("Butterworth Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

plt.xlim(0, 200)
plt.grid()

plt.show()