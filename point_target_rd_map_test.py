from simulator import PDRadarSimulator, generate_tx_chirp
from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import scipy

c = scipy.constants.c 

# radar config
radar_config = {
    # data acq parameters
    'PRI': 1.0541e-3, # s PRI
    'PRF': 948.68, # Hz, PRF
    'M_cpi': 128, # pulses per CPI
    'CPI':134.92e-3, # s, CPI duration

    # signal parameters
    'B': 14.99e6, # Hz, chirp bandwidth
    'T_chirp': 105.41e-6, # s, chirp duration
    'fs': 22.48e6, # Hz, baseband sampling rate
    'lambda_':0.1999, # m, wavelength
    'fc': 1.5e9, # Hz, center frequency

    # radar performance parameters
    'r_max': 158e3, # m, max unambiguous range
    'dr': 10, # m, range resolution 
    'v_max': 47.4, # m/s, max unambiguous velocity    
    'dv': 0.74, # m/s, velocity resolution
}



# generate chirp signal, s_full
s_full, tx_template, N_p, N_pri = generate_tx_chirp(fs=radar_config['fs'], B=radar_config['B'], T_p=radar_config['T_chirp'], T_pri=radar_config['PRI'])

pd_radar_simulator = PDRadarSimulator(
    fc=radar_config['fc'],
    T_pri=radar_config['PRI'],
    fs=radar_config['fs'],  
    M_slow=radar_config['M_cpi'],
    tx_full=s_full,
    tx_template=tx_template
)

# add scatterers
pd_radar_simulator.add_scatterer(
    r=40e3, # m
    v=15, # m/s
    rcs=1, # m^2
    phi=0 # rad, initial doppler phase
)

pd_radar_simulator.add_scatterer(
    r=60e3, # m
    v=13, # m/s
    rcs=1, # m^2
    phi=0 # rad, initial doppler phase
)

# generate echo matrix
echo_matrix = pd_radar_simulator.generate_echo_matrix()

fig, ax = plt.subplots()
cax = ax.imshow(np.abs(echo_matrix), aspect='auto', cmap='jet', origin='lower')
fig.colorbar(cax)
plt.title("Echo Matrix | Amplitude")
plt.xlabel("Fast Time Sample Index")
plt.ylabel("Slow Time Index (Pulse Number)")
plt.tight_layout()
plt.show()


# perform MF to each row of the echo matrix
mf_output = pd_radar_simulator.matched_filtering_fft(echo_matrix)

# 距离轴
M, N = mf_output.shape
rng_axis = np.arange(N) * c / (2 * radar_config['fs']) # in m
rng_axis = rng_axis / 1e3  # 转换为 km

# 绘图
plt.figure(figsize=(10, 6))
epsilon = 1e-12
amp_dB = 20 * np.log10(np.abs(mf_output) + epsilon)
plt.imshow(amp_dB, aspect='auto', cmap='jet', origin='lower',
           extent=[rng_axis[0], rng_axis[-1], 0, M-1])  # extent 显式标注坐标轴范围

plt.xlabel("Range (m)")
plt.ylabel("Pulse Index")
plt.title("Matched Filter Output")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()


# doppler process
rd_map = pd_radar_simulator.doppler_process(mf_output, window=True)

# 绘图
plt.figure(figsize=(10, 6))
epsilon = 1e-12
amp_dB = 20 * np.log10(np.abs(rd_map) + epsilon)

plt.imshow(amp_dB, aspect='auto', cmap='jet', origin='lower',
           extent=[rng_axis[0], rng_axis[-1], 0, M-1])  # extent 显式标注坐标轴范围

plt.xlabel("Range (m)")
plt.ylabel("Pulse Index")
plt.title("RD Map")
plt.colorbar(label="Amplitude")
plt.tight_layout()
plt.show()
