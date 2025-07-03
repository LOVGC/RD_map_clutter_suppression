import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt

# ===== Step 1: Continuous signal parameters =====
f1 = 20  # Hz
f2 = 31  # Hz
fs = 80  # Sampling frequency
T = 1    # Duration in seconds
t = np.arange(0, T, 1/fs)  # Sampling time vector

# ===== Step 2: Create continuous-time signal x(t) =====
x = 2*np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# ===== Step 3: Compute FFT and IFFT =====
X =  fft(x)
X_shift = fftshift(X)  # Shift zero frequency component to center
# The  fftfreq function is designed to return the sample frequencies corresponding to the output of  fft
freq = fftfreq(len(t), d=1/fs)  
freq_shift = fftshift(freq) 

x_recon = ifft(X)
x_recon_shift = ifft(X_shift)  # Reconstructed signal from shifted FFT

# ===== Step 4: Plotting =====
plt.figure(figsize=(14, 10))

# Time-domain signal
plt.subplot(4, 1, 1)
plt.plot(t, x, label='Original x(t)')
plt.title("Original Sampled Signal in Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Frequency-domain signal (magnitude spectrum)
plt.subplot(4, 1, 2)
plt.stem(freq_shift, np.abs(X_shift), basefmt=" ")
plt.title("Magnitude Spectrum |X(f)|")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

# Reconstructed signal after IFFT
plt.subplot(4, 1, 3)
plt.plot(t, x_recon.real, label='Reconstructed x(t)', linestyle='--')
plt.title("Reconstructed Signal from X")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Reconstructed signal after IFFT
plt.subplot(4, 1, 4)
plt.plot(t, x_recon_shift.real, label='Reconstructed x(t)', linestyle='--')
plt.title("Reconstructed Signal from X_shift")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
