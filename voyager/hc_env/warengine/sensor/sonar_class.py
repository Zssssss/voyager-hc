"""
# python
# -*- coding:utf-8 -*-
@Project : SQ
@File : sonar_class.py
@Author : HuangHaoning
@Time : 2024/3/5 15:24
@Description :
"""
import copy
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
from scipy.fftpack import fft, rfft, rfftfreq
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84
from pyproj import CRS

crs_WGS84 = CRS.from_epsg(4326)
from pyproj import Transformer


def lla_to_xyz(lat, lon, alt):
    transprojr = Transformer.from_crs(
        "EPSG:4326",
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True)
    x, y, z = transprojr.transform(lon, lat, alt, radians=False)
    return np.array([x, y, z])


def xyz_to_lla(x, y, z):
    transprojr = Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        "EPSG:4326",
        always_xy=True)
    lon, lat, alt = transprojr.transform(x, y, z, radians=False)
    return np.array([lat, lon, alt])


def theta_to_angle(theta):
    theta = theta * 180 / np.pi
    if theta <= 90:
        theta = 90 - theta
    elif theta <= 270:
        theta = -(theta - 90)
    else:
        theta = 450 - theta
    return theta


class Propagation_Model:
    def __init__(self):
        self.I0 = 0.67 * 10 ** (-18)
        # (W/M^2)介质参考声强,在水声学中定义为均方根声压为1 μPa (微帕)的平面波的声强，≈0.67×10^{-18} W/m^2。
        self.alpha0 = 0.01  # 海面的声吸收系数
        self.IN = 0.83 * 10 ** (-17)  # 带宽内噪声强度
        self.N0 = 0.1  # 1Hz带宽内噪声功率
        self.n = 1  # 声波类型不同，n不同，

    def get_sl(self, P):
        '''
        声源级
        P:(W)为无指向性声源的辐射声功率
        I0:(W/M^2)介质参考声强,在声学中定义为1×10^{-12} W/m^2，
        I0在水声学中定义为均方根声压为1 μPa (微帕)的平面波的声强，≈0.67×10^{-18} W/m^2。
        '''
        I = P / 4 / np.pi  # 距离声源1m处的声强
        # SL=10*np.log10(P)+170.77#水声学
        SL = 10 * np.log10(I / self.I0)
        return SL

    def get_tl(self, r, h):
        '''
        传播损失
        r:(m)距离
        h:(m)水深
        alpha0:海面的声吸收系数
        n:声波类型不同，n不同，
        n = 1，平面波，无扩展损失；
        n = 2，柱面波，无界面吸收的声传播，如全反射海底、海面；
        n = 3，柱面波，计入界面吸收的声传播；
        n = 4，球面波
        n = 5，通过浅海负跃变层的声传播损失；
        n = 6，球面波，计入平整海面的声反射干涉，也适用于声源辐射声场远场衰减。
        '''
        alpha = self.alpha0 * (1 - 6.67 * 10 ** (-5) * (-h))
        # 为h的声吸收系数。深度增加1000m，吸收系数减少6.67%。
        TL = self.n * 10 * np.log10(r) + alpha * r
        return TL

    def get_ts(self, target_density):
        '''
        目标强度
        当声波遇到目标物（如潜艇或鱼群）时，会产生回声。
        目标强度或回声强度表示目标对声波的反射能力
        描述目标反射能力的大小，与声波特性（频率、波形）和目标特性（形状、材料、密度）等有关。
        表现为距目标1m处，反射平面波强度与入射平面波强度的比值；
        '''
        TS = 1 * target_density
        return TS

    def get_nl(self):
        '''
        环境噪声级
        IN:带宽内噪声强度
        I0:参考声强。
        '''
        NL = 10 * np.log10(self.IN / self.I0)
        return NL

    def get_rl(self, I):
        '''
        等效平面波混响级
        I0:参考声强。
        I:与混响场等效的平面波强度
        '''
        RL = 10 * np.log10(I / self.I0)
        return RL

    def get_dt(self, S, T):
        '''
        检测阈值
        是对于预定置信级下，接收机输入端所需要的接收带宽（或1Hz带宽）内信号功率与噪声功率之比。
        DT值越低，说明设备的处理能力越强。
        N0: 1Hz带宽内噪声功率
        S:信号功率
        T:信号脉冲宽
        '''
        d = 2 * S * T / (self.N0)
        DT = 10 * np.log10(d / (2 * T))
        return DT

    def judge_activate_sonar(self, SL, TL, TS, NL, RL, DT):
        if SL - 2 * TL + TS - NL - RL >= DT:
            return True, SL - 2 * TL + TS - NL - RL
        else:
            return False, 0

    def judge_sonar(self, SL, TL, TS, NL, RL, DT):
        if SL - TL - NL >= DT:
            return True, SL - TL - NL
        else:
            return False, 0


class Target:
    """"初始化各个声源（潜艇、渔船、海浪等）的属性，如频率、速度、声功率、密度、经纬度、海拔、名字"""

    def __init__(self, sub_lat, sub_lon, sub_alt, v, f=10000, P=10, target_density=0, sub_name=""):
        self.f = f  # 频率，HZ
        self.v = v
        self.P = P
        self.target_density = target_density
        self.sub_lat = sub_lat
        self.sub_lon = sub_lon
        self.sub_alt = sub_alt
        self.name = sub_name

    def pos_distribution(self, ORIGIN_pos, scale):
        # 建模声源可传播的声场空间
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0
        z_min = 0  # add
        z_max = 0  # add
        source_pos = lla_to_xyz(self.sub_lat, self.sub_lon, self.sub_alt)
        r = np.sqrt(np.sum((ORIGIN_pos - source_pos) * (ORIGIN_pos - source_pos)))
        print("目标距离：", r)
        source_pos -= ORIGIN_pos
        ORIGIN_pos -= ORIGIN_pos
        source_pos /= scale
        r = np.sqrt(np.sum((ORIGIN_pos - source_pos) * (ORIGIN_pos - source_pos)))
        print("以声呐为坐标原点，scale=", scale, "  目标坐标：", source_pos, "目标距离：", r * scale)
        if source_pos[0] <= x_min:
            x_min = int(source_pos[0]) - 1
            x_max = scale
        else:
            x_max = int(source_pos[0]) + 1
            x_min = -scale
        if source_pos[1] <= y_min:
            y_min = int(source_pos[1]) - 1
            y_max = scale
        else:
            y_max = int(source_pos[1]) + 1
            y_min = -scale
        if source_pos[2] <= z_min:
            z_min = int(source_pos[2]) - 1
            z_max = scale
        else:
            z_max = int(source_pos[2]) + 1
            z_min = -scale

        x = np.arange(x_min, x_max + 1, 1)
        y = np.arange(y_min, y_max + 1, 1)
        z = np.arange(z_min, z_max + 1, 1)  # add
        xx, yy, zz = np.meshgrid(x, y, z)  # add
        # print(x_min,x_max,y_min,y_max,z_min,z_max)
        r = np.sqrt((xx * scale - source_pos[0] * scale) ** 2 +
                    (yy * scale - source_pos[1] * scale) ** 2 +
                    (zz * scale - source_pos[2] * scale) ** 2)
        r[r < 1] = 1

        return xx, yy, zz, r  # add


class Environment:
    # 环境类，负责对声源信息建模、声传播，并反馈给声呐
    def __init__(self, subs):
        self.subs = subs
        self.c = 1500  # 波速
        self.alpha = 0.1
        # self.alpha0 = 0.03
        # self.I0 = 0.67 * 10 ** (-18)
        # self.n = 1
        self.propagation_model = Propagation_Model()

    def cal_p(self, P, r, h, target_density=0):
        # 计算功率为P的声源，距离r处声强级，h为声源深度
        # I = P / 4 / np.pi  # 距离声源1m处的声强
        # SL = 10 * np.log10(I / I0)
        SL = self.propagation_model.get_sl(P)
        # alpha = alpha0 * (1 - 6.67 * 10 ** (-5) * h)  # 为h的声吸收系数。深度增加1000m，吸收系数减少6.67%。
        # TL = n * 10 * np.log10(r) + alpha * r
        TL = self.propagation_model.get_tl(r, h)
        TS = self.propagation_model.get_ts(target_density)
        p = -TL + SL + TS

        if type(p) == np.float64:
            p = max(0, p)
        else:
            p[p < 0] = 0
        return p

    def get_passive_info(self, sonar_v, sonar_lat, sonar_lon, sonar_alt, drag_array):
        # 建模环境中的被动信号（对声纳来说是被动）
        # s = time.time()
        receiver_pos = lla_to_xyz(sonar_lat, sonar_lon, sonar_alt)
        vec_norm = np.linalg.norm(receiver_pos, axis=0)
        receiver_dir = receiver_pos / vec_norm  # 归一化接收器方向
        sonar_receive_p = []
        i = 0
        # print("\n被动信号，开始环境建模*************************")
        for sub in self.subs:
            i += 1
            source_pos = lla_to_xyz(sub.sub_lat, sub.sub_lon, sub.sub_alt)
            r = np.sqrt(np.sum((receiver_pos - source_pos) * (receiver_pos - source_pos)))
            p = self.cal_p(sub.P, r, sub.sub_alt) / (r ** self.alpha)
            f2 = self.Doppler_effect(sonar_v, sub.v, self.c, sub.f)
            # print(r)

            vec_norm = np.linalg.norm(source_pos - receiver_pos, axis=0)
            source_dir = (source_pos - receiver_pos) / vec_norm  # 归一化声波方向
            cos_theta = np.sum(receiver_dir * source_dir)  # 夹角余弦值
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            cos_theta = max(abs(cos_theta), abs(sin_theta))  # 按xyz坐标系的角度
            if drag_array:  # 拖曳声呐阵元探测能力更好
                lat = sub.sub_lat - sonar_lat + random.uniform(-0.000001, 0.000001)
                lon = sub.sub_lon - sonar_lon + random.uniform(-0.000001, 0.000001)
            else:
                lat = sub.sub_lat - sonar_lat + random.uniform(-0.0001, 0.0001)
                lon = sub.sub_lon - sonar_lon + random.uniform(-0.0001, 0.0001)
            # print(lat,lon)
            theta = np.arctan(lat / lon)  # 按经纬度的方位,arctan返回范围在-0.5pi至0.5pi
            if lon >= 0 and lat <= 0:
                theta = 2 * np.pi + theta
            if lon <= 0:
                theta = np.pi + theta
            sonar_receive_p.append([p, f2, cos_theta, theta])
            # 111000m=1lat，误差10m左右就是0.0001lat
        # e=time.time()
        # print(e - s)
        return sonar_receive_p

    def Doppler_effect(self, sonar_v, sub_v, v, f):  # 多普勒频移
        f2 = (v + sonar_v) / (v + sub_v) * f
        # f2=f2+random.uniform(-10,10)
        # print(v,sonar_v,sub_v,f,f2)
        return f2

    def func1(self, r):
        return self.propagation_model.n * 10 * np.log10(r) + \
               self.propagation_model.alpha0 * r - self.sonar_SL

    def emit_and_receive(self, lat, lon, alt, sonar_v, emit_dir, emit_P, emit_f, emit_angle):
        # 对主动声呐发射的声波进行传播、建模，并反馈给主动声呐
        # print("\n主动声呐发射信号,环境建模****************************")
        # 计算发射波声强级，
        SL = self.propagation_model.get_sl(emit_P)
        # I = emit_P / 4 / np.pi  # 距离声源1m处的声强
        # SL = 10 * np.log10(I / self.propagation_model.I0)#声源处声强级
        # print("声源处声强级SL=",SL)
        # 估算最大可探测距离
        self.sonar_SL = SL
        max_r = fsolve(self.func1, 1)[0] / 2
        # print("可探测最大距离max_r=",max_r)
        # 建模声呐传播空间，判断有无target在范围内，有的话，计算回弹和两段衰减
        # 计算目标位置和发射角度夹角及距离是否符合
        sonar_pos = lla_to_xyz(lat, lon, alt)
        Rebounds = []
        for sub in self.subs:
            sub_pos = lla_to_xyz(sub.sub_lat, sub.sub_lon, sub.sub_alt)
            real_dir = sub_pos - sonar_pos
            real_dir = real_dir / np.linalg.norm(real_dir, axis=0) + [random.uniform(-0.01, 0.01) for ii in
                                                                      range(0, 3)]  # 加上方位噪声
            theta = np.arccos(np.sum((sub_pos - sonar_pos) * emit_dir) /
                              (np.linalg.norm((sub_pos - sonar_pos), axis=0) *
                               np.linalg.norm(emit_dir, axis=0))) / np.pi * 180
            sub_r = np.sqrt(np.sum((sub_pos - sonar_pos) * (sub_pos - sonar_pos)))
            # print(sub.name,"发射器偏角=",theta,"距离=",sub_r)
            # if sub_r>max_r or theta>emit_angle:#不在探测范围内
            #     continue
            if sub_r > max_r:  # 不在探测范围内
                continue
            # print(sub.name,"在探测范围内")
            rebound_time = 2 * sub_r / self.c
            rebound_f = self.Doppler_effect(sonar_v, sub.v, self.c, emit_f)
            rebound_p = self.cal_p(emit_P, 2 * sub_r, sub.sub_alt, sub.target_density)
            # print(rebound_p)
            if rebound_p <= 0:
                continue
            Rebounds.append(
                {"rebound_time": rebound_time, "rebound_f": rebound_f, "rebound_p": rebound_p, "dir": real_dir})
        return Rebounds


class passive_sonar:  # 被动声呐类
    def __init__(self, sonar_lat, sonar_lon, sonar_alt, v, env, name="声呐", drag_array=False):
        self.sonar_lat = sonar_lat
        self.sonar_lon = sonar_lon
        self.sonar_alt = sonar_alt
        self.v = v
        self.DT = 20  # 接收阈值
        self.env = env
        self.name = name
        self.drag_array = drag_array

    def fft_trans(self, y, N, plt_show=False):  # 快速傅里叶变换函数
        fft_y = fft(y)  # 快速傅里叶变换
        x = np.arange(N)  # 频率个数
        half_x = x[range(int(N / 2))]  # 取一半区间

        abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
        angle_y = np.angle(fft_y)  # 取复数的角度
        normalization_y = abs_y / N  # 归一化处理（双边频谱）
        normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
        if not plt_show:
            return half_x, normalization_half_y
        # plt.subplot(231)
        # plt.plot(x[0:50], y[0:50])
        # plt.title('原始波形')
        #
        # plt.subplot(232)
        # plt.plot(x, fft_y, 'black')
        # plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')
        #
        # plt.subplot(233)
        # plt.plot(x, abs_y, 'r')
        # plt.title('双边振幅谱(未归一化)', fontsize=9, color='red')
        #
        # plt.subplot(234)
        # plt.plot(x, angle_y, 'violet')
        # plt.title('双边相位谱(未归一化)', fontsize=9, color='violet')
        #
        # plt.subplot(235)
        # plt.plot(x, normalization_y, 'g')
        # plt.title('双边振幅谱(归一化)', fontsize=9, color='green')
        #
        # plt.subplot(236)
        # plt.plot(half_x, normalization_half_y, 'blue')
        # plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
        #
        # plt.show()

    def add_noisy(self, signal_y):  # 声波加噪
        max_y = np.max(signal_y)
        noise = np.random.normal(max_y / 10, 3, signal_y.shape)
        return signal_y + noise

    # def gen_sonar_img(self, signals):  # 生成被动声纳水平方向能量图
    #     point_num = 360
    #     r = np.zeros(point_num)
    #     thetas = np.linspace(0, np.pi * 2, point_num)
    #     s = np.pi * 2 / point_num
    #     for signal in signals:
    #         energy = signal["p_recevied"]
    #         angle = signal["theta"]
    #         if angle > 2 * np.pi:
    #             print("error")
    #         i = round(angle / s) % point_num
    #         r[i] += energy
    #     fig = plt.figure()
    #     ax = plt.subplot(projection='polar')
    #     width = np.pi * 2 / point_num
    #     ax.bar(thetas, r, width=width)
    #     plt.title(self.name + "噪声水平能量分布图")
    #     plt.show()

    def passive_detect(self, active=False, plt_show=False, env=None):  # 被动声呐探测函数
        '''
        被动接收海洋环境中的声波，输出信号列表，每个信号特征如下：
        {"theta": theta, "f": f, "p_recevied": p_recevied}
        信号方位角、频率、信号声强级
        '''
        s = time.time()
        # 环境为声呐提供声场信号
        if env is not None:
            self.env = env
        sonar_receive_p = self.env.get_passive_info(self.v, self.sonar_lat, self.sonar_lon, self.sonar_alt,
                                                    self.drag_array)
        # print("\n"+self.name+"   开始建模被动信号*************************")
        # 接收处理声场信号
        signals = []
        res = []
        min_f = 999999999
        max_f = -999999999
        f = None
        for i in range(0, len(sonar_receive_p)):
            receive_p = sonar_receive_p[i]
            p0 = receive_p[0]  # 声纳处声强级
            if p0 < self.DT:  # 检测阈值
                continue
            f = receive_p[1]
            cos_theta = receive_p[2]
            theta = receive_p[3]

            p_recevied = p0 * cos_theta
            if p_recevied <= 0:
                p_recevied = 0
                continue
            T = 1 / f
            I0 = 10 ** (p_recevied / 10) * self.env.propagation_model.I0  # 声呐处声强
            P0 = I0 * 4 * np.pi  # 声呐处接收功率

            # print("方位=",theta / np.pi * 180,"频率=", f," 接收声强级=",p_recevied)
            if f <= min_f:
                min_f = f
            if f >= max_f:
                max_f = f
            bias = random.uniform(0, T)
            signals.append({"theta": theta, "f": f, "T": T, "P0": P0, "p_recevied": p_recevied, "bias": bias})
            res.append({"theta": theta, "f": round(f, 3), "p_recevied": round(p_recevied, 3)})
        if f is None:
            return None
        if active:
            return min_f, max_f, signals

        N = int(5 * max_f)
        # print(N)
        signal_x = np.linspace(0, 1, N)
        signal_y = 0 * signal_x
        for signal in signals:
            signal_y += np.log10(signal["p_recevied"]) * \
                        np.sin(2 * np.pi * signal["f"] * signal_x + signal["bias"])
        signal_y = self.add_noisy(signal_y)
        half_x, normalization_half_y = self.fft_trans(signal_y, N)

        if not plt_show:
            return res
        else:
            return (res, signal_x, signal_y)
        # 绘制接收声波图像
        # self.gen_sonar_img(signals)
        # plt.subplot(211)
        # plt.plot(signal_x[0:int(10 / min_f * N)], signal_y[0:int(10 / min_f * N)], linewidth=1)
        # plt.title(self.name + '接收原始波形', fontsize=9, color='blue')
        # plt.subplot(212)
        # plt.plot(half_x, normalization_half_y, linewidth=1)
        # plt.title(self.name + '功率谱密度', fontsize=9, color='blue')
        # plt.show()

        # return res


class active_sonar(passive_sonar):  # 主动声呐类
    def __init__(self, sonar_lat, sonar_lon, sonar_alt, dir, v, env, f, P, angle, name="声呐"):
        passive_sonar.__init__(self, sonar_lat, sonar_lon, sonar_alt, v, env, name)
        self.v = v
        self.f = f
        self.lat = sonar_lat
        self.lon = sonar_lon
        self.alt = sonar_alt
        self.emit_dir = dir
        self.emit_P = P
        self.angle = angle
        self.emit_dir = self.emit_dir / np.linalg.norm(self.emit_dir, axis=0)

    def cal_target_speed(self, receive_f):
        c = self.env.c
        sub_v = (c + self.v) / (receive_f / self.f) - c
        return sub_v

    def Emit_And_Receive(self):
        return self.env.emit_and_receive(self.lat, self.lon, self.alt, self.v, self.emit_dir,
                                         self.emit_P, self.f, self.angle)

    def acive_detect(self, plt_show=True):  # 主动声呐探测函数
        '''
               主动发射并接收回波信号，输出信号列表，每组信号特征如下：
               {"f": rebound_f, "p": rebound_p,"bias":rebound_time,"pos":(lat,lon,alt),"r":sub_r}
               信号频率、信号声强级、回波时间、目标经纬度和海拔、目标距离
        '''
        receive_info = self.Emit_And_Receive()  # 发射声波并接收回弹脉冲
        sonar_pos = lla_to_xyz(self.lat, self.lon, self.alt)
        # min_f,max_f,passive_signals=self.passive_detect(active=True)#被动干扰信号
        min_f = 999999999
        max_f = -999999999
        signals = []
        # for signal in passive_signals:
        #    if signal["p_recevied"]<self.DT:#检测阈值
        #        continue
        #    signals.append({"f":signal["f"],"p":signal["p_recevied"],"bias":signal["bias"]})
        # print("\n"+self.name+"  开始建模主动信号*************************")
        rebound_f = None
        for info in receive_info:
            rebound_time = round(info["rebound_time"], 3)
            rebound_f = round(info["rebound_f"], 3)
            rebound_p = round(info["rebound_p"], 3)
            sub_dir = info["dir"]
            sub_r = round(self.env.c * rebound_time / 2, 3)
            sub_pos = sonar_pos + sub_r * sub_dir
            # print(sub_dir,sub_r,sub_pos-sonar_pos,np.sqrt(np.sum((sub_pos-sonar_pos)*(sub_pos-sonar_pos))))
            lat, lon, alt = xyz_to_lla(sub_pos[0], sub_pos[1], sub_pos[2])
            lat = round(lat, 5)
            lon = round(lon, 5)
            alt = round(alt, 3)
            I0 = 10 ** (rebound_p / 10) * self.env.propagation_model.I0  # 声呐处声强
            P0 = I0 * 4 * np.pi  # 声呐处接收功率
            if max_f < rebound_f:
                max_f = rebound_f
            signals.append({"f": rebound_f, "p": rebound_p, "bias": rebound_time, "pos": (lat, lon, alt), "r": sub_r})
        if rebound_f is None:
            return None

        if not plt_show:
            return signals
        N = int(3 * max_f)
        # print(N)
        signal_x = np.linspace(0, 1, N)
        signal_y = 0 * signal_x
        for signal in signals:
            # signal_y += 10**12*signal["P0"] * np.sin(2 * np.pi * signal["f"] * signal_x)
            signal_y += np.log10(signal["p"]) * np.sin(2 * np.pi * signal["f"] * signal_x + signal["bias"])
        signal_y = self.add_noisy(signal_y)

        if plt_show:
            return (signals, signal_x, signal_y)
        # # 绘制接收声波图像
        # plt.subplot(211)
        # # print(max_f,min_f)
        # # plt.plot(signal_x[0:int(10 / min_f * N)], signal_y[0:int(10 / min_f * N)])
        # plt.plot(signal_x[0:int(0.2 * N)], signal_y[0:int(0.2 * N)], linewidth=0.5)
        # plt.title('主动声呐接收原始波形', fontsize=9, color='blue')
        # plt.subplot(212)
        # half_x, normalization_half_y = self.fft_trans(signal_y, N)
        # plt.plot(half_x, normalization_half_y, linewidth=2)
        # plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
        # plt.show()
        # return signals

    # def acive_detect_all_dir(self):#主动声呐全方位旋转探测


class passive_sonar_combination:  # 多个被动声呐定位模型
    def __init__(self, sonars):
        self.sonars = sonars

    def solve_eqs(self, e1, e2):
        # 求交点
        if e1["b"] == e2["b"]:
            return -1, -1
        lon = (e1["b"] * e1["lon"] - e2["b"] * e2["lon"] + e2["lat"] - e1["lat"]) / (e1["b"] - e2["b"])
        lat = e1["lat"] - e1["b"] * (e1["lon"] - lon)
        return lon, lat

    def analyse_location_type(self, plt_show=True):
        '''
        多个被动声呐接收海洋环境中的声波，并分析给出各个目标的位置：纬度、经度
        '''
        fre2equ = {}
        for sonar in self.sonars:
            signal = sonar.passive_detect(plt_show=plt_show)
            if signal is not None:
                if plt_show:
                    signals = signal[0]
                else:
                    signals = signal
                if len(signals) < 1:  # 此声呐未接受到有效信号
                    continue
                signals.sort(key=lambda x: x['f'], reverse=True)  # 信号按频率级倒序排序
                for s in signals:
                    # print("方位=", s["theta"] / np.pi * 180, "频率=", s["f"], " 接收声强级=", s["p_recevied"])
                    if s["f"] not in fre2equ:
                        fre2equ[s["f"]] = []
                    fre2equ[s["f"]].append({"lat": sonar.sonar_lat, "lon": sonar.sonar_lon, "b": math.tan(s["theta"])})
            else:
                continue
        res = []
        # 定位
        for f in fre2equ:
            if len(fre2equ[f]) < 2:
                continue
            equations = fre2equ[f]
            res_lat = []
            res_lon = []
            for i in range(0, min(len(equations), 3)):
                for j in range(i + 1, min(len(equations), 3)):
                    e1 = equations[i]
                    e2 = equations[j]
                    lon, lat = self.solve_eqs(e1, e2)
                    if lon == -1 and lat == -1:
                        continue
                    res_lat.append(lat)
                    res_lon.append(lon)
            if len(res_lat) == 0:
                continue
            lat = round(sum(res_lat) / len(res_lat), 5)
            lon = round(sum(res_lon) / len(res_lon), 5)
            res.append({"lat": lat, "lon": lon, "f": f})
        return res


class Passive_drag_array_sonar:  # 被动拖曳阵声呐
    def __init__(self, env, rope_len, hydrophone_num, hydrophone_dis):
        '''
        线形阵
        JT_lat,JT_lon,JT_alt:舰艇位置
        v:舰艇速度
        rope_len:绳缆长度
        hydrophone_num：阵元个数
        hydrophone_dis：阵元间隔
        '''
        self.env = env
        self.rope_len = rope_len
        self.hydrophone_num = hydrophone_num
        self.hydrophone_dis = hydrophone_dis
        self.hydrophones = []
        for i in range(0, hydrophone_num):
            # 计算声呐位置
            hydrophone = passive_sonar(sonar_lat=12, sonar_lon=120, sonar_alt=0,
                                       v=0, env=env, drag_array=True)
            self.hydrophones.append(hydrophone)

    def solve_eqs(self, e1, e2):
        # print(e1,e2)
        if e1["b"] == e2["b"]:
            return -1, -1
        lon = (e1["b"] * e1["lon"] - e2["b"] * e2["lon"] + e2["lat"] - e1["lat"]) / (e1["b"] - e2["b"])
        lat = e1["lat"] - e1["b"] * (e1["lon"] - lon)
        return lon, lat

    def analyse_location(self, JT_lat, JT_lon, JT_alt, v_lon, v_lat, theta_rope, theta_hydrophone,
                         thermocline_height=-90, plt_show=False):
        '''
        以D型缆阵为例，其主要技术参数包括拖缆直径、长度和重量，以及拖曳阵的直径、长度和重量。
        具体来说，拖缆直径可能为9.5mm，长度达到800m，重量为205kg；
        而拖曳阵直径可能是82.5mm（也有报道为89mm），长度为75m，重量为640kg。
        JT_lat,JT_lon,JT_alt: 舰艇位置
        v_lon,v_lat: 舰艇速度分量
        theta_rope:绳缆与海平面夹角
        theta_hydrophone：拖曳阵与绳缆夹角
        theta_rope+theta_hydrophone<90度
        '''
        # 建模阵元位置
        theta_rope = theta_rope / 180 * math.pi
        theta_hydrophone = theta_hydrophone / 180 * math.pi
        theta = np.arctan(abs(JT_lon / JT_lat))  # JT所处位置地心连线、经线之间的夹角大小
        v = np.sqrt(v_lon ** 2 + v_lat ** 2)
        dir_lon = -v_lon / abs(v_lon)
        dir_lat = -v_lat / abs(v_lat)
        Delta_h1 = self.rope_len * math.sin(theta_rope)
        Delta_lat1 = self.rope_len * math.sin(theta + theta_rope)
        Delta_lon1 = self.rope_len * math.cos(theta + theta_rope)
        lat_sonar = []
        lon_sonar = []
        alt_sonar = []
        for i in range(0, len(self.hydrophones)):
            Delta_h2 = self.hydrophone_dis * i * math.sin(theta_rope + theta_hydrophone)
            Delta_lat2 = self.hydrophone_dis * i * math.sin(theta + theta_rope + theta_hydrophone)
            Delta_lon2 = self.hydrophone_dis * i * math.cos(theta + theta_rope + theta_hydrophone)
            Delta_h = Delta_h1 + Delta_h2
            Delta_lat = (Delta_lat1 + Delta_lat2) / 111320
            Delta_lon = (Delta_lon1 + Delta_lon2) / 111320
            alt_i = JT_alt - Delta_h
            lat_i = JT_lat + dir_lat * Delta_lat
            lon_i = JT_lon + dir_lon * Delta_lon
            lat_sonar.append(lat_i)
            lon_sonar.append(lon_i)
            alt_sonar.append(alt_i)
            self.hydrophones[i].sonar_lat = lat_i
            self.hydrophones[i].sonar_lon = lon_i
            self.hydrophones[i].sonar_alt = alt_i
            self.hydrophones[i].v = v
            self.hydrophones[i].name = "阵元" + str(i)

        # 跃变层信息处理
        hydrophones_alt = np.mean(alt_sonar)
        # print('hydrophones_alt',hydrophones_alt)
        env = []
        for i in range(len(self.env.subs)):
            if (abs(hydrophones_alt) < abs(thermocline_height) and abs(self.env.subs[i].sub_alt) < abs(
                    thermocline_height)) or (
                    abs(hydrophones_alt) >= abs(thermocline_height) and abs(self.env.subs[i].sub_alt) >= abs(
                    thermocline_height)):
                env.append(self.env.subs[i])
        self.env.subs = env

        # 每个阵元接收信号并处理
        fre2equ = {}
        flag = False
        feature = {'pos': {f'sonar{i}': {} for i in range(len(self.hydrophones))},
                   'feature_info': {f'sonar{i}': [] for i in range(len(self.hydrophones))}}
        if plt_show:
            data = {f'sonar{i}': {"x": [], "y": []} for i in range(len(self.hydrophones))}
        i = 0
        for hydrophone in self.hydrophones:
            signal = hydrophone.passive_detect(plt_show=plt_show, env=self.env)
            if signal is not None:
                if plt_show:
                    signals = signal[0]
                    feature['feature_info']["sonar{}".format(i)] = signals
                    feature["pos"]["sonar{}".format(i)] = {"lat": lat_sonar[i], "lon": lon_sonar[i],
                                                           "height": alt_sonar[i]}
                    data["sonar{}".format(i)]['x'] = signal[1]
                    data["sonar{}".format(i)]['y'] = signal[2]
                else:
                    signals = signal
                    feature['feature_info']["sonar{}".format(i)] = signals
                    feature["pos"]["sonar{}".format(i)] = {"lat": lat_sonar[i], "lon": lon_sonar[i],
                                                           "height": alt_sonar[i]}
                i += 1
                if len(signals) < 1:  # 此声呐未接受到有效信号
                    continue
            else:
                continue
            flag = True
            signals.sort(key=lambda x: x['f'], reverse=True)  # 信号按频率级倒序排序
            for s in signals:
                # print("方位=", s["theta"] / np.pi * 180, "频率=", s["f"], " 接收声强级=", s["p_recevied"])
                if s["f"] not in fre2equ:
                    fre2equ[s["f"]] = []
                fre2equ[s["f"]].append(
                    {"lat": hydrophone.sonar_lat, "lon": hydrophone.sonar_lon, "b": math.tan(s["theta"])})
        if not flag:
            return False

        # 定位
        res = []
        for f in fre2equ:
            if len(fre2equ[f]) < 2:
                continue
            equations = fre2equ[f]
            res_lat = []
            res_lon = []
            for i in range(0, min(len(equations), 3)):
                for j in range(i + 1, min(len(equations), 3)):
                    e1 = equations[i]
                    e2 = equations[j]
                    lon, lat = self.solve_eqs(e1, e2)
                    if lon == -1 and lat == -1:
                        continue
                    res_lat.append(lat)
                    res_lon.append(lon)
            if len(res_lat) == 0:
                continue
            lat = round(sum(res_lat) / len(res_lat), 5)
            lon = round(sum(res_lon) / len(res_lon), 5)
            res.append({"lat": lat, "lon": lon, "f": f})
        if plt_show:
            result = {'target_pos': res, "sonar_info": {"feature": feature, 'data': data}}
        else:
            result = {'target_pos': res, "sonar_info": {"feature": feature}}

        return result


'''
调用示例：
'''


def run_env():
    start = time.time()
    # 按需求生成多个声源，包括潜艇、渔船、海浪。
    sub1 = Target(sub_lat=12.02, sub_lon=120, sub_alt=-100, v=-30, f=1000, P=50, target_density=10,
                  sub_name="潜艇")  # v:靠近声呐方向为-，远离为+
    sub2 = Target(sub_lat=15.015, sub_lon=120, sub_alt=-100, v=-5, f=200, P=1, target_density=5, sub_name="渔船")
    subs = [sub1, sub2]

    # for i in range(0, 1):
    #     # 生成频率为20-3.15khz的海洋噪声
    #     f = random.uniform(20, 3150)
    #     v = random.uniform(0, 20)
    #     sub_lat = random.uniform(11.95, 12.05)
    #     sub_lon = random.uniform(119.95, 120.05)
    #     sub_alt = random.uniform(-100, 0)
    #     sub = Target(sub_lat=sub_lat, sub_lon=sub_lon, sub_alt=sub_alt, v=v, f=f, P=10,
    #                  target_density=1, sub_name="海洋噪声" + str(i))
    #     subs.append(sub)
    # 2. 初始化声场环境，加入上述声源
    env = Environment(subs)
    return env


def run_one_passive_sonar(env):
    # 被动声呐探测目标
    sonar = passive_sonar(sonar_lat=14.97, sonar_lon=120.005, sonar_alt=-100, v=0, env=env, name="被动1")
    print('12', geod.Inverse(14.97, 120.005,15.02, 120)['s12'])
    res = sonar.passive_detect(plt_show=True)  # 单个被动声呐
    # for info in res[0]:
    #     print("方位=", theta_to_angle(info["theta"]) ,info["theta"]*180/np.pi, "频率=", info["f"], " 接收声强级=", info["p_recevied"])
    end = time.time()
    # print("time:", end - start)
    return res


def run_one_active_sonar(env):
    # 主动动声呐探测目标
    sonar = active_sonar(sonar_lat=15.07, sonar_lon=120.005, sonar_alt=0, dir=[75.06, -21.11, 214.31],
                         v=0, env=env, f=2000, P=1, angle=10, name="主动1")
    # print('s12', geod.Inverse(12.07, 120.005, 12.02, 120)['s12'])
    # print('theta', geod.Inverse(12.02, 120.005, 12.02, 120)['azi1'])
    res = sonar.acive_detect(plt_show=True)
    # for sub_info in res:
    #     print("频率=", sub_info["f"], " 声强级=", sub_info["p"], "信号回传时间=", sub_info["bias"],
    #           "经纬海拔=", sub_info["pos"], "预估距离=", sub_info["r"])
    return res


def run_passive_sonar_combination(env):
    # 1. 三个被动动声组成的呐阵列
    start = time.time()
    sonar1 = passive_sonar(sonar_lat=11.97, sonar_lon=120.005, sonar_alt=0, v=0, env=env, name="被动1")
    sonar2 = passive_sonar(sonar_lat=12.01, sonar_lon=119.98, sonar_alt=0, v=0, env=env, name="被动2")
    # sonar3 = passive_sonar(sonar_lat=11.99, sonar_lon=120.02, sonar_alt=0, v=0, env=env, name="被动3")
    # sonar4 = passive_sonar(sonar_lat=15.021, sonar_lon=120.02, sonar_alt=0, v=0, env=env, name="被动4")
    # sonar5 = passive_sonar(sonar_lat=15.022, sonar_lon=120.02, sonar_alt=0, v=0, env=env, name="被动3")
    # sonar6 = passive_sonar(sonar_lat=15.019, sonar_lon=120.02, sonar_alt=0, v=0, env=env, name="被动3")

    passive_sonars = [sonar1, sonar2]
    sonar_combination = passive_sonar_combination(passive_sonars)
    # 2. 探测目标
    res_sonar_combination = sonar_combination.analyse_location_type(plt_show=True)  # plt_show=False时延低
    if not res_sonar_combination:
        print("没有可确定位置的目标")
    else:
        print("\n声呐阵列定位结果如下：************************")
        for info in res_sonar_combination:
            print(info)
    end = time.time()
    print("time:", end - start)
    return res_sonar_combination


def run_passive_drag_array_sonar(env):
    # 被动拖曳阵声呐探测目标
    rope_len = 800
    theta_rope = 10
    theta_hydrophone = 20
    Drag_array_sonar = Passive_drag_array_sonar(env, rope_len=rope_len, hydrophone_num=3, hydrophone_dis=2)
    res_Drag_array = Drag_array_sonar.analyse_location(JT_lat=15.03, JT_lon=120.01, JT_alt=0, v_lon=10, v_lat=-30,
                                                       theta_rope=theta_rope, theta_hydrophone=theta_hydrophone, thermocline_height=-90,
                                                       plt_show=False)
    print('12',geod.Inverse(15.03,120.01,15.02,120)['s12'])
    # for i in range(len(res_Drag_array['sonar_info']['feature'])):
    #     info = res_Drag_array['sonar_info']['feature']
    # for j in range(len(info)):
    #     info[j]['theta'] = theta_to_angle(info[j]['theta'])
    # print('12', geod.Inverse(res_Drag_array['sonar_info']['feature']['sonar{}'.format(i)]['pos']['lat'], res_Drag_array['sonar_info']['feature']['sonar{}'.format(i)]['pos']['lon'], 12.02, 120)['azi1'])
    # print('123', res_Drag_array)
    # if not res_Drag_array:
    #     print(False)
    # else:
    #     print(True)

    return res_Drag_array

    # print("\n被动拖曳阵声呐定位结果如下：************************")
    # for info in res_Drag_array:
    #     print(info)


def main():
    env = run_env()
    # a = run_one_passive_sonar(env)
    # a = run_one_active_sonar(env)
    a = run_passive_sonar_combination(env)
    # a = run_passive_drag_array_sonar(env)
    return a

if __name__ == '__main__':
    start = time.time()
    a = main()
    print(a)
    print(time.time()-start)
