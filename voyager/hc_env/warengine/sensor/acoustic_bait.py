import numpy as np
import matplotlib.pyplot as plt
import math
from geographiclib.geodesic import Geodesic
import warnings
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

geod = Geodesic.WGS84


class acoustic_bait_Model:
    def __init__(self):
        self.source_level = 200  # 发射声源级,单位dB
        self.noise_fre = 50  # 单位kHz
        self.sea_state = 3
        self.absorption_coefficient = 13  # 吸收系数
        self.target_strength = 10  # 潜艇目标强度(声诱饵模拟目标强度)
        self.beam_width = 5
        self.detection_threshold = 20  # 检测阀，越大噪声越大，但是探测方位会变广
        self.fc = 11_000  # 载波频率

    # def

    def gen_bait(self, bait_lat, bait_lon, detector_lat, detector_lon, sea_state=1, bait_speed=10, detect_speed=10,
                 sensor_img=False):
        R = geod.Inverse(detector_lat, detector_lon, bait_lat, bait_lon)['s12']

        Qs = [89, 281.8, 707.9, 891.3, 1259, 1500, 1778.3]
        f = float(self.noise_fre) * 1000
        noise_level = 20 * math.log10(2000 * Qs[int(sea_state)] / f + f * 5.6234 / 30000)
        directivity_index = 10 * math.log10(36000 / float(self.beam_width) ** 2)  # 计算指向性指数
        transmission_loss = lambda r: 20 * np.log10(r) + float(self.absorption_coefficient) * r / 1000  # 根据距离计算传播损失
        echo_level = lambda r: float(self.source_level) - 2 * transmission_loss(r) + float(
            self.target_strength)  # 根据距离计算回声
        noise_masking_level = noise_level - directivity_index + float(self.detection_threshold)  # 计算噪音屏蔽水平
        attract = False

        # 回声余量
        A = echo_level(R) - noise_masking_level  # 根据A的大小可以判断是否能发现潜艇，若A＞0表示能发现（若声诱饵模拟的目标强度可以使A>0，才能够成功诱骗地方）

        v_sonar = 1530  # 声波在海水中的传播速度是1530m/s
        v_detect = detect_speed  # 观察者的速度，如飞机的声呐浮标
        v_acoustic = bait_speed  # 声诱饵的速度
        save_file_base64 = None
        if A > 0:
            # 可以接收到声诱饵的信号
            attract = True
            if sensor_img:
                td = R / v_sonar  # 时延，单位秒
                fd = (v_sonar + v_detect) * self.fc / (v_sonar - v_acoustic)  # 多普勒频移，单位Hz
                zeta = 0.1  # 回波展宽,声波在传播后返回发射源时(由于水体对声波的吸收、散射和反射作用造成的)出现信号扩散的现象，从而使反射回来的信号强度减弱
                pulse_width = 5e-3  # 脉宽为5ms

                t = np.linspace(0, 0.2, 1000)
                # S = np.cos(2 * np.pi * fc * t)
                S = 0.3 * np.cos(2 * np.pi * self.fc * t)
                S[0:int(self.fc * pulse_width)] = 10 * np.exp(1j * self.fc * t[0:int(self.fc * pulse_width)])  # 入射波信号
                delay_s = np.concatenate([np.random.uniform(-0.3, 0.3, int(self.fc * td)), S])
                # delay_t = np.concatenate([np.zeros((int(fc * td))), t[0:int(fc * pulse_width)]])
                # delay_t = np.concatenate([delay_t, np.ones((len(delay_s) - len(delay_t)))*t[1]])
                e = 10 ** (self.target_strength / 20) * delay_s * np.exp(1j * (self.fc - fd)) * np.exp(-1j * zeta)
                t = np.linspace(0, 0.2 * len(e) / len(t), len(e))

                # plt.plot(S)
                # plt.plot(delay_s)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(t, e)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                save_file = BytesIO()
                plt.savefig(save_file, format="png")
                save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
        else:
            # 声呐浮标不能接收到声诱饵的信号
            attract = False
            if sensor_img:
                t = np.linspace(0, 0.2, 1000)
                # S = np.cos(2 * np.pi * fc * t)
                S = 0.3 * np.random.randn(len(t)) * np.exp(1j * self.fc * t)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(t, S)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                save_file = BytesIO()
                plt.savefig(save_file, format="png")
                save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')

        return save_file_base64, attract


if __name__ == '__main__':
    import time

    start_time = time.time()
    bait = acoustic_bait_Model()
    save_file_base64, attract = bait.gen_bait(detector_lat=13, detector_lon=120, bait_lat=12.001, bait_lon=120)
    end_time = time.time()
    print('total time: {}'.format(end_time - start_time))

# R = 1000
# source_level = 200  # 发射声源级,单位dB
# frequency = 50  # 单位kHz
# sea_state = 3
# absorption_coefficient = 13  # 吸收系数
# target_strength = 10  # 潜艇目标强度(声诱饵模拟目标强度)
# beam_width = 5
# detection_threshold = 20  # 检测阀，越大噪声越大，但是探测方位会变广
# fc = 11_000  # 载波频率
# test = 1
#
# # 根据海况和频率计算噪声级
# Qs = [89, 281.8, 707.9, 891.3, 1259, 1500, 1778.3]
# f = float(frequency) * 1000
# noise_level = 20 * math.log10(2000 * Qs[int(sea_state)] / f + f * 5.6234 / 30000)
# directivity_index = 10 * math.log10(36000 / float(beam_width) ** 2)  # 计算指向性指数
# transmission_loss = lambda r: 20 * np.log10(r) + float(absorption_coefficient) * r / 1000  # 根据距离计算传播损失
# echo_level = lambda r: float(source_level) - 2 * transmission_loss(r) + float(target_strength)  # 根据距离计算回声
# noise_masking_level = noise_level - directivity_index + float(detection_threshold)  # 计算噪音屏蔽水平
#
# # 回声余量
# A = echo_level(R) - noise_masking_level  # 根据A的大小可以判断是否能发现潜艇，若A＞0表示能发现（若声诱饵模拟的目标强度可以使A>0，才能够成功诱骗地方）
# if test:
#     x = np.linspace(0.1, 3000, 10000)
#     el = echo_level(x)
#     nml = np.full_like(x, noise_masking_level)
#
#     # Calculate where lines cross
#     eq_point = np.where(np.isclose(el, nml, 1e-3))[0][0]
#     eq_point_m = x[eq_point]
#     eq_point_db = el[eq_point]
#
#     fig, ax = plt.subplots()
#     plt.title("Echo Level and Noise Masking Level vs. Range")
#     ax.plot(x, el, label="Echo Level")
#     ax.plot(x, nml, label="Noise Masking Level")
#     ax.plot(eq_point_m, eq_point_db, 'ro',
#             label=f"Equivalence Point ({round(eq_point_db, 1)} dB @ {round(eq_point_m, 1)} meters)")
#     ax.set_xlabel('Range (m)')
#     ax.set_ylabel('Sound Pressure Level (dB re 1 uPa)')
#     ax.legend()
#     plt.ylim(0, 200)
#     plt.show()
#
# v_sonar = 1530  # 声波在海水中的传播速度是1530m/s
# v_detect = 10  # 观察者的速度，如飞机的声呐浮标
# v_acoustic = 10  # 声诱饵的速度
# if A > 0:
#     # 可以接收到声诱饵的信号
#     td = R / v_sonar  # 时延，单位秒
#     fd = (v_sonar + v_detect) * fc / (v_sonar - v_acoustic)  # 多普勒频移，单位Hz
#     zeta = 0.1  # 回波展宽,声波在传播后返回发射源时(由于水体对声波的吸收、散射和反射作用造成的)出现信号扩散的现象，从而使反射回来的信号强度减弱
#     pulse_width = 5e-3  # 脉宽为5ms
#
#     t = np.linspace(0, 0.2, 1000)
#     # S = np.cos(2 * np.pi * fc * t)
#     S = 0.3 * np.cos(2 * np.pi * fc * t)
#     S[0:int(fc * pulse_width)] = 10 * np.exp(1j * fc * t[0:int(fc * pulse_width)])  # 入射波信号
#     delay_s = np.concatenate([np.random.uniform(-0.3, 0.3, int(fc * td)), S])
#     # delay_t = np.concatenate([np.zeros((int(fc * td))), t[0:int(fc * pulse_width)]])
#     # delay_t = np.concatenate([delay_t, np.ones((len(delay_s) - len(delay_t)))*t[1]])
#     e = 10 ** (target_strength / 20) * delay_s * np.exp(1j * (fc - fd)) * np.exp(-1j * zeta)
#     t = np.linspace(0, 0.2 * len(e) / len(t), len(e))
#
#     # plt.plot(S)
#     # plt.plot(delay_s)
#     plt.plot(t, e)
#     plt.show()
# else:
#     # 声呐浮标不能接收到声诱饵的信号
#     t = np.linspace(0, 0.2, 1000)
#     # S = np.cos(2 * np.pi * fc * t)
#     S = 0.3 * np.random.randn(len(t)) * np.exp(1j * fc * t)
#     plt.plot(S)
#     plt.show()
