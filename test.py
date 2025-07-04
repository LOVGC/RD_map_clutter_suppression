from simulator import PDRadarSimulator, generate_tx_chirp
from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt
import numpy as np


# radar config
fc = 1.5e9 # center freq
T_pri = 5e-3 # PRI
fs = 30e6  # baseband sampling rate

# chirp config
T_p = 10e-6  # Chirp duration
B = 20e6  # Chirp bandwidth


s_full, tx_template, N_p, N_pri = generate_tx_chirp(fs=1e6, B=20e6, T_p=0.01, T_pri=0.1)




fs = 40e6
B = 20e6
T_p = 10e-6
T_pri = 50e-6

s_full, tx_chirp, N_p, N_PRI = generate_tx_chirp(fs, B, T_p, T_pri)

# 画 chirp 的频谱
f = fftshift(fftfreq(N_p, 1/fs))
S = fftshift(fft(tx_chirp))

plt.plot(f/1e6, 20*np.log10(np.abs(S)))
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.title("Chirp Spectrum (Center Freq ≈ 0 Hz)")
plt.grid(True)
plt.show()
