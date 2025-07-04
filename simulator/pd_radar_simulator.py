import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal.windows import hann
import scipy
from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.pyplot as plt

def generate_tx_chirp(fs, B, T_p, T_pri):
    """
    生成一个 PRI 长度的 chirp 信号（baseband），前 T_p 为 chirp，后面补零。
    返回值:
        s_full: 长度为 T_pri 的完整 chirp 信号（含零）
        tx_chirp: 原始 chirp 模板（长度为 T_p）
        N_p: chirp 点数
        N_PRI: PRI 点数
    """
    N_p = int(round(T_p * fs))      # Chirp 长度（点数）
    N_pri = int(round(T_pri * fs))  # PRI 总长度（点数）

    t = np.arange(N_p) / fs
    mu = B / T_p  # 调频率

    tx_chirp = np.exp(1j * np.pi * mu * t**2)
    s_full = np.zeros(N_pri, dtype=complex)
    s_full[:N_p] = tx_chirp

    return s_full, tx_chirp, N_p, N_pri


class PDRadarSimulator:
    def __init__(self, fc, T_pri, fs, M_slow, tx_full, tx_template):
        """
        初始化 PD 雷达模拟器
        fc: 载频 Hz
        T_pri: pulse repetition interval (秒)
        fs: baseband 采样率 Hz
        M_slow: 脉冲个数
        tx_full: 完整的发射 chirp 信号（PRI 长度）
        tx_template: 单个脉冲的 chirp 模板（长度为 T_p）, T_p 是脉冲持续时间
        """
        self.fc = fc
        self.T_pri = T_pri
        self.fs = fs
        self.N = int(round(T_pri * fs))  # 每个 T_pri 内的采样点数
        self.M = M_slow # 脉冲个数
        self.c = scipy.constants.c 
        self.scatterers = []
        self.tx_full = tx_full
        self.tx_template = tx_template

    def add_scatterer(self, r, v, rcs, phi):
        """
        添加一个散射中心（距离[m], 速度[m/s], RCS/幅度）
        """
        self.scatterers.append({'r': r, 'v': v, 'rcs': rcs, 'doppler_init_phase':phi})

    def _generate_scatterer_echo_matrix(self, scatterer):
        """
        为单个散射中心生成回波（M x N）
        """
        r0 = scatterer['r']
        v = scatterer['v']
        alpha = scatterer['rcs']
        phi = scatterer['doppler_init_phase']
        tau = 2 * r0 / self.c
        f_D = 2 * v * self.fc / self.c

        t_slow = np.arange(self.M) * self.T_pri # slow time 时间采样点
        echo_matrix = np.zeros((self.M, self.N), dtype=complex)

        # 生成延迟版 chirp（chirp 头部移位，前面补零）
        n_delay = int(round(tau * self.fs))     # 延迟对应的采样点数

        if n_delay >= self.N - len(self.tx_template):
            print(f"Scatterer at {r0} m with delay {n_delay} exceeds PRI length {self.N - len(self.tx_template)}. Skipping.")
            return echo_matrix  # 超过 PRI 就不处理了（或者可拓展跨 PRI 功能),return 一个都是0的 echo matrix.

        tx_delayed = np.roll(self.tx_full, n_delay)  # roll 是循环 shift, 但是这里因为 tx_full 后面是0，所以相当于 shift 后，前面补零
        if n_delay > 0:
            tx_delayed[:n_delay] = 0  # 这个就强制了前面补零

        # loop over 每一个延迟后的脉冲，给每一个脉冲加上 Doppler phase
        for m in range(self.M): 
            doppler_phase = np.exp(1j * 2 * np.pi * f_D * t_slow[m] + phi)  # 计算第 m 个脉冲的 Doppler phase
            echo_matrix[m, :] = alpha * doppler_phase * tx_delayed

        return echo_matrix

    # todo: 需要加 window 吗？
    def matched_filtering_corr(self, echo_matrix):
        """
        对每个脉冲（行）进行 fast-time 匹配滤波
        """
        M, N = echo_matrix.shape
        mf_output = np.zeros_like(echo_matrix, dtype=complex)

        mf_kernel = np.conj(self.tx_template[::-1])  # 匹配滤波器

        for m in range(M):
            conv_result = correlate(echo_matrix[m, :], mf_kernel, mode='same')
            mf_output[m, :] = conv_result

        return mf_output
    
    def matched_filtering_fft(self, echo_matrix):
        """
        频域匹配滤波（循环相关），每个脉冲（行）做 fast-time matched filter。
        """
        M, N = echo_matrix.shape
        H = np.conj(fft(self.tx_template, n=N))  # 匹配滤波核：conj(s[n])
        R = fft(echo_matrix, axis=1)
        Y = R * H[None, :]
        mf_output = ifft(Y, axis=1)
        return mf_output  # 注意这是 circular correlation, 有可能有问题

    def generate_echo_matrix(self):
        """
        对所有目标生成总的 echo_matrix（叠加）
        """
        total_echo_matrix = np.zeros((self.M, self.N), dtype=complex)
        for scatterer in self.scatterers:
            total_echo_matrix += self._generate_scatterer_echo_matrix(scatterer)
        return total_echo_matrix
    
    def generate_rd_map(self, mf_output, window=True):
        """
        对匹配滤波后的信号做 slow-time FFT，得到 RD 图
        mf_output: 匹配滤波后的 echo_matrix，形状 M x N
        """
        if window:
            win = hann(self.M)[:, None]
            mf_output = mf_output * win

        rd_map = fftshift(fft(mf_output, axis=0), axes=0)
        rd_map = np.abs(rd_map)
        return rd_map

    # toDo: 这个 vmax 是干嘛的？
    def plot_rd_map(self, rd_map, vmax=None, title="RD Map"):
        """
        绘制 RD 图，Doppler 轴用速度（m/s）表示
        rd_map: 2D array, complex valued
        """
        M, N = rd_map.shape

        # Doppler frequency axis → velocity axis (in m/s)
        f_doppler = fftshift(fftfreq(M, self.T_pri))
        v_axis = f_doppler * self.c / (2 * self.fc)

        # Range axis
        rng_axis = np.arange(N) * (self.c / (2 * self.fs))

        plt.figure(figsize=(10, 5))
        plt.imshow(np.abs(rd_map), aspect='auto',
                extent=[rng_axis[0], rng_axis[-1], v_axis[0], v_axis[-1]],
                origin='lower', cmap='jet', vmax=vmax)

        plt.xlabel("Range (m)")
        plt.ylabel("Velocity (m/s)")
        plt.title(title)
        plt.colorbar(label="Amplitude")
        plt.tight_layout()
        plt.show()


