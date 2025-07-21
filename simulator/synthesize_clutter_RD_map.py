import numpy as np
import scipy
import matplotlib.pyplot as plt


c = scipy.constants.c  # 光速，单位 m/s

def generate_tx_chirp(fs, B, T_chirp, T_pri):
    """
    生成一个 PRI 长度的chirp信号，其中前 T_p 部分为chirp，后面补零
    fs: baseband sampling frequency, Hz
    B: chirp bandwidth, Hz
    T_chirp: chirp pulse duration, seconds
    T_pri: pulse repetition interval, seconds
    """
    N_p = int(round(T_chirp * fs))      # 脉冲内采样点数
    N_PRI = int(round(T_pri * fs))    # PRI内总采样点数

    t = np.arange(N_p) / fs
    t = t - T_chirp / 2  # centered frequency 

    mu = B / T_chirp
    tx_chirp = np.exp(1j * np.pi * mu * t**2)

    s_full = np.zeros(N_PRI, dtype=complex)
    s_full[:N_p] = tx_chirp
    return s_full, tx_chirp, N_p, N_PRI


def simulate_clutter_echo_IQ(M, fs, B, T_p, T_pri, clutter_rmax, K=100):
    """
    模拟 M 个脉冲，每个脉冲持续 T_p，PRI 为 T_pri，包含 K 个散射点的杂波IQ信号矩阵
    clutter_rmax: 最大杂波距离，单位 m
    返回：
        C: M × N_PRI 矩阵，每行是一个 pulse 内接收的IQ信号
    """
    # 生成带 dead time 的发射 chirp 信号
    s_full, tx_chirp, N_p, N_PRI = generate_tx_chirp(fs, B, T_p, T_pri) # N_p: 每个 pulse(chirp) 的采样点数，N_PRI: 每个 PRI 的采样点数

    # 最大延迟 clutter_rmax 对应的时间延迟
    tau_max = 2 * clutter_rmax / c   # 单位：秒

    # 初始化接收信号矩阵
    C = np.zeros((M, N_PRI), dtype=complex)

    # 模拟 K 个散射点
    for _ in range(K):
        A_k = np.sqrt(np.random.exponential())      # 振幅 √指数分布（Rayleigh envelope）
        phi_k = np.random.uniform(0, 2 * np.pi)     # 初始相位 （这个感觉有问题）
        f_D_k = np.random.normal(0, 1 / (10 * T_pri)) # 多普勒频率，近似0Hz左右
        tau_k = np.random.uniform(0, tau_max)           # 延迟，0~T_p 之间

        n_delay = int(round(tau_k * fs))            # 延迟对应的采样点数

        if n_delay >= N_PRI - N_p:
            continue  # 超过 PRI 就不处理了（或者可拓展跨 PRI 功能）

        # 生成延迟版 chirp（chirp 头部移位，前面补零）
        s_delayed = np.roll(s_full, n_delay)
        if n_delay > 0:
            s_delayed[:n_delay] = 0

        # 累加 K 个散射点回波
        for m in range(M):
            doppler_phase = np.exp(1j * (2 * np.pi * f_D_k * m * T_pri + phi_k))
            C[m, :] += A_k * doppler_phase * s_delayed

    return C, tx_chirp, N_p, N_PRI

def compute_RD_map(C, tx_chirp, N_PRI):
    # 获取匹配滤波器
    
    H = np.conj(np.fft.fft(tx_chirp, n=N_PRI))
    # 距离压缩（频域匹配滤波），注意，这里严格来讲是循环卷积，不是线性卷积，但是最后结果是差别不大的。
    R = np.fft.ifft(np.fft.fft(C, axis=1) * H[None, :], axis=1)

    # 多普勒处理（慢时间 FFT）
    RD = np.fft.fftshift(np.fft.fft(R, axis=0), axes=0)

    return RD, R 




fs = 10e6         # 10 MHz
B = 5e6           # 5 MHz
T_p = 20e-6       # 20 us
T_pri = 100e-6    # 100 us (10kHz PRF)
M = 128           # 128 pulses per CPI
K = 500           # 500 clutter scatterers
clutter_rmax = 2000  # 3 km clutter范围

C, tx_chirp, N_p, N_PRI = simulate_clutter_echo_IQ(M, fs, B, T_p, T_pri, clutter_rmax, K)

RD, R = compute_RD_map(C, tx_chirp, N_PRI)

###############画图##############

fig, axs = plt.subplots(2, 1, figsize=(12, 8))

# 计算真实距离坐标
range_axis = np.arange(N_PRI) * c / (2 * fs)

# ===== R 图 =====
im1 = axs[0].imshow(np.abs(R), aspect='auto', cmap='jet',
                    extent=[range_axis[0], range_axis[-1], 0, M])
axs[0].set_title("Range Compressed Data (R)")
axs[0].set_ylabel("Pulse Index")
axs[0].set_xlabel("Range (meters)")
fig.colorbar(im1, ax=axs[0], orientation='vertical', label='Amplitude')

# ===== RD 图 =====
im2 = axs[1].imshow(20 * np.log10(np.abs(RD) + 1e-6), aspect='auto',
                    cmap='jet',
                    extent=[range_axis[0], range_axis[-1], -M//2, M//2])
axs[1].set_title("Range-Doppler Map (RD)")
axs[1].set_ylabel("Doppler Bin")
axs[1].set_xlabel("Range (meters)")
fig.colorbar(im2, ax=axs[1], orientation='vertical', label='Magnitude (dB)')

plt.tight_layout()
plt.show()
