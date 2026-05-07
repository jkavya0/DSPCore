import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (butter, filtfilt)

fs = 1000                     # Sampling frequency
t = np.linspace(0, 1, fs)

# Mixed signal
x = (np.sin(2*np.pi*5*t) +
     0.5*np.sin(2*np.pi*50*t) +
     0.3*np.sin(2*np.pi*120*t))

# Low-pass filter
fc = 20
b, a = butter(N=4,Wn=fc/(fs/2),btype='low')
y_low = filtfilt(b, a, x)

# High-pass filter
fc = 40
b, a = butter(N=4,Wn=fc/(fs/2),btype='high')
y_high = filtfilt(b, a, x)

# Band-pass filter
lowcut = 40
highcut = 80
b, a = butter( N=4,Wn=[lowcut/(fs/2), highcut/(fs/2)],btype='band')
y_band = filtfilt(b, a, x)
plt.figure(figsize=(14,10))

# Original
plt.subplot(4,1,1)
plt.plot(t, x)
plt.title("Original Mixed Signal")
plt.grid()

# Low-pass
plt.subplot(4,1,2)
plt.plot(t, y_low)
plt.title("Low Pass Filter Output")
plt.grid()

# High-pass
plt.subplot(4,1,3)
plt.plot(t, y_high)
plt.title("High Pass Filter Output")
plt.grid()

# Band-pass
plt.subplot(4,1,4)
plt.plot(t, y_band)
plt.title("Band Pass Filter Output")
plt.grid()

plt.tight_layout()
plt.show()