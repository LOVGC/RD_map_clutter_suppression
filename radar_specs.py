import numpy as np
import scipy.constants

def r_v_paras_to_signal_paras(r_max, dr, v_max, dv, fc, duty_cycle=0.1):
    """
    计算脉冲多普勒雷达的关键设计参数
    
    参数:
    r_max     : float - 最大无模糊距离 (m)
    dr        : float - 距离分辨率 (m)
    v_max     : float - 最大无模糊速度 (m/s)
    dv        : float - 速度分辨率 (m/s)
    fc        : float - 载波频率 (Hz)
    duty_cycle: float, optional - 发射占空比 (默认0.1)
    
    返回:
    dict - 包含雷达参数的字典:
        PRI       : 脉冲重复间隔 (s)
        B         : 调频带宽 (Hz)
        T_chirp   : 线性调频持续时间 (s)
        M_cpi     : 一个CPI内的脉冲数
        fs        : 快时间采样率 (Hz)
        PRF       : 脉冲重复频率 (Hz)
        f_nyquist : 多普勒奈奎斯特频率 (Hz)
        T_cpi     : 相关处理间隔时间 (s)
        lambda_   : 雷达波长 (m)
        dr_actual : 实际距离分辨率 (m)
        dv_actual : 实际速度分辨率 (m/s)
        r_max_actual : 实际最大无模糊距离 (m)
        v_max_actual : 实际最大无模糊速度 (m/s)
        duty_cycle: 占空比
    """
    c = scipy.constants.c  # 光速 (m/s)
    lambda_ = c / fc  # 波长 (m)
    
    # 1. 检查基本参数有效性
    if not all(x > 0 for x in [r_max, dr, v_max, dv, fc]):
        raise ValueError("所有输入参数必须为正数")
    if not (0 < duty_cycle < 1):
        raise ValueError("占空比必须在(0,1)范围内")
    
    # 2. 计算带宽 (距离分辨率约束)
    B = c / (2 * dr)
    
    # 3. 计算PRF范围 (无模糊距离和速度约束)
    PRF_min = 4 * v_max / lambda_  # 速度约束要求的最小PRF
    PRF_max = c / (2 * r_max)      # 距离约束要求的最大PRF
    
    # 检查参数冲突
    if PRF_min > PRF_max:
        required_condition = 8 * v_max * r_max <= c * lambda_
        raise ValueError(f"参数冲突: 要求PRF>{PRF_min:.2f}Hz(速度约束)且PRF<{PRF_max:.2f}Hz(距离约束). "
                         f"必须满足 8*v_max*r_max <= c*lambda_ ({required_condition})")
    
    # 4. 选择PRF (取约束范围内的中间值)
    PRF = np.sqrt(PRF_min * PRF_max)  # 几何平均平衡两个约束
    PRI = 1 / PRF

    # 计算实际的最大无模糊距离和速度
    r_max_actual = c / (2 * PRF)  # 实际最大无模糊距离
    v_max_actual = (lambda_ * PRF) / 4  # 实际最大无模糊速度
    
    # 5. 计算脉冲宽度和占空比
    T_chirp = duty_cycle * PRI
    
    # 6. 计算所需脉冲数 (速度分辨率约束)
    T_cpi = lambda_ / (2 * dv)       # 所需相关处理间隔时间
    M_cpi_float = T_cpi / PRI        # 理论脉冲数
    
    # 取整为2的幂 (便于FFT处理)
    M_cpi = int(2**np.ceil(np.log2(M_cpi_float)))
    
    # 7. 计算快时间采样率 (带宽约束)
    fs = 1.5 * B  # 20%裕量满足采样定理
    
    # 8. 计算实际分辨率
    dr_actual = c / (2 * B)
    dv_actual = lambda_ / (2 * M_cpi * PRI)
    
    return {
        'PRI': PRI,
        'B': B,                   # waveform bandwidth
        'T_chirp': T_chirp,
        'M_cpi': M_cpi,           # 一个CPI内的脉冲数
        'fs': fs,                 # 快时间采样率, at baseband
        'PRF': PRF,
        'f_nyquist': PRF / 2,     # 多普勒奈奎斯特频率
        'T_cpi': M_cpi * PRI,     # 实际相关处理间隔时间
        'lambda_': lambda_,
        'dr_actual': dr_actual,   # 实际距离分辨率
        'dv_actual': dv_actual,   # 实际速度分辨率
        'r_max_actual': r_max_actual,  # 实际最大无模糊距离
        'v_max_actual': v_max_actual,  # 实际最大无模糊速度
        'duty_cycle': duty_cycle
    }

def signal_paras_to_r_v_paras(PRI, B, fc, M_cpi):
    """
    根据雷达信号参数计算雷达性能参数
    
    参数:
        PRI: 脉冲重复间隔 (s)
        B: 调频带宽 (Hz)
        fc: 载波频率 (Hz)
        M_cpi: 一个CPI内的脉冲数
        
    返回:
        dict: 包含雷达性能参数的字典:
            r_max: 最大无模糊距离 (m)
            dr: 距离分辨率 (m)
            v_max: 最大无模糊速度 (m/s)
            dv: 速度分辨率 (m/s)
    """
    c = scipy.constants.c  # 光速 (m/s)
    PRF = 1 / PRI  # 脉冲重复频率 (Hz)
    lambda_ = c / fc  # 波长 (m)

    r_max = c * PRI / 2  # 最大无模糊距离 (m)
    dr = c / (2 * B)  # 距离分辨率 (m)
    v_max = (lambda_ * PRF) / 4  # 最大无模糊速度 (m/s)
    dv = lambda_ / (2 * M_cpi * PRI)  # 速度分辨率 (m/s)
    
    return {
        'r_max': r_max,
        'dr': dr,
        'v_max': v_max,
        'dv': dv
    }

# 示例用法: 改动 input_params 的值来测试不同的参数组合
if __name__ == "__main__":
    try:
        # 定义输入参数 (典型值)
        input_params = {
            'r_max': 100e3,   # m 最大无模糊距离
            'dr': 10,         # m 距离分辨率
            'v_max': 30,      # m/s 最大无模糊速度
            'dv': 1,          # m/s 速度分辨率
            'fc': 1.5e9,      # Hz 载频
            'duty_cycle': 0.1 # 10%占空比
        }
        
        print("输入参数:")
        print(f"r_max = {input_params['r_max']/1000:.1f} km")
        print(f"dr = {input_params['dr']} m")
        print(f"v_max = {input_params['v_max']} m/s")
        print(f"dv = {input_params['dv']} m/s")
        print(f"fc = {input_params['fc']/1e9:.1f} GHz")
        print(f"duty_cycle = {input_params['duty_cycle']*100:.0f}%")
        print("\n" + "="*50 + "\n")
        
        # 通过性能参数计算信号参数
        signal_paras = r_v_paras_to_signal_paras(**input_params)
        
        # 打印信号参数
        print("计算得到的信号参数:")
        print(f"PRI = {signal_paras['PRI']*1e3:.4f} ms")
        print(f"PRF = {signal_paras['PRF']:.2f} Hz ({signal_paras['PRF']/1e3:.2f} kHz)")
        print(f"B = {signal_paras['B']/1e6:.2f} MHz")
        print(f"M_cpi = {signal_paras['M_cpi']}")
        print(f"T_chirp = {signal_paras['T_chirp']*1e6:.2f} μs")
        print(f"fs = {signal_paras['fs']/1e6:.2f} MHz")
        print(f"λ = {signal_paras['lambda_']:.4f} m")
        print(f"实际距离分辨率 = {signal_paras['dr_actual']:.2f} m")
        print(f"实际速度分辨率 = {signal_paras['dv_actual']:.2f} m/s")
        print(f"实际最大无模糊距离 = {signal_paras['r_max_actual']/1000:.2f} km")
        print(f"实际最大无模糊速度 = {signal_paras['v_max_actual']:.2f} m/s")
        print(f"相关处理间隔时间(T_cpi) = {signal_paras['T_cpi']*1e3:.2f} ms")
        print(f"多普勒奈奎斯特频率 = ±{signal_paras['f_nyquist']:.1f} Hz")
        print("\n" + "="*50 + "\n")
        
        # 使用信号参数反向计算性能参数
        rv_paras = signal_paras_to_r_v_paras(
            PRI=signal_paras['PRI'],
            B=signal_paras['B'],
            fc=input_params['fc'],
            M_cpi=signal_paras['M_cpi']
        )
        
        # 打印反向计算的性能参数
        print("反向计算的性能参数:")
        print(f"r_max = {rv_paras['r_max']/1000:.2f} km (原始输入: {input_params['r_max']/1000:.1f} km)")
        print(f"dr = {rv_paras['dr']:.2f} m (原始输入: {input_params['dr']} m)")
        print(f"v_max = {rv_paras['v_max']:.2f} m/s (原始输入: {input_params['v_max']} m/s)")
        print(f"dv = {rv_paras['dv']:.2f} m/s (原始输入: {input_params['dv']} m/s)")
        print("\n" + "="*50 + "\n")
        
        # 验证反向计算的一致性
        print("一致性验证:")
        # 1. 距离分辨率应完全匹配
        assert np.isclose(rv_paras['dr'], input_params['dr']), f"距离分辨率不匹配: {rv_paras['dr']} vs {input_params['dr']}"
        print("✅ 距离分辨率匹配")
        
        # 2. 实际最大无模糊距离应与反向计算的r_max匹配
        assert np.isclose(rv_paras['r_max'], signal_paras['r_max_actual']), \
            f"最大无模糊距离不匹配: {rv_paras['r_max']} vs {signal_paras['r_max_actual']}"
        print("✅ 最大无模糊距离匹配")
        
        # 3. 实际最大无模糊速度应与反向计算的v_max匹配
        assert np.isclose(rv_paras['v_max'], signal_paras['v_max_actual']), \
            f"最大无模糊速度不匹配: {rv_paras['v_max']} vs {signal_paras['v_max_actual']}"
        print("✅ 最大无模糊速度匹配")
        
        # 4. 速度分辨率可能有微小差异（由于M_cpi取整）
        dv_diff = abs(rv_paras['dv'] - signal_paras['dv_actual'])
        dv_tolerance = 0.01  # 1% 容差
        if dv_diff > dv_tolerance:
            print(f"⚠️ 速度分辨率差异较大: {rv_paras['dv']:.4f} vs {signal_paras['dv_actual']:.4f}")
        else:
            print("✅ 速度分辨率匹配")
        
        # 5. 验证输入v_max和实际v_max的关系
        if signal_paras['v_max_actual'] < input_params['v_max']:
            print(f"⚠️ 注意: 实际最大无模糊速度({signal_paras['v_max_actual']:.2f} m/s) "
                  f"小于输入值({input_params['v_max']} m/s)")
        else:
            print("✅ 实际最大无模糊速度满足要求")
        
        # 6. 验证输入r_max和实际r_max的关系
        if signal_paras['r_max_actual'] < input_params['r_max']:
            print(f"⚠️ 注意: 实际最大无模糊距离({signal_paras['r_max_actual']/1000:.2f} km) "
                  f"小于输入值({input_params['r_max']/1000:.1f} km)")
        else:
            print("✅ 实际最大无模糊距离满足要求")
        
    except ValueError as e:
        print("参数错误:", e)