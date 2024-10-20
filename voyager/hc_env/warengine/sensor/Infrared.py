"""
@Project: Infrared simulation
@File   : Infrared.py
@Describe :对红外传感器进行仿真，需要考虑渔船、鱼群、潜艇、货轮的影响
"""
import random

import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
import time
import warnings
import scipy.integrate as integ
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84

warnings.filterwarnings("ignore")


def lla_to_xyz(lat, lon, alt):
    transprojr = Transformer.from_crs(
        "EPSG:4326",
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True)
    x, y, z = transprojr.transform(lon, lat, alt, radians=False)
    return x, y, z


class Submarine:
    def __init__(self):
        self.len = 66.8  # 潜艇的长度，单位m，参考美国长颌须鱼号潜艇
        self.wid = 8.8  # 潜艇的宽度，单位m


class infrared:
    def __init__(self):
        self.image_height = 60
        self.image_width = 60
        self.focal_leng = 8e-3  # 焦距为8mm
        self.theta = 87 * np.pi / 180  # 视场角为30度
        self.proportion = self.focal_leng * np.tan(self.theta) / (
            np.sqrt((self.image_height // 2) ** 2 + (self.image_width // 2) ** 2))

    def mapping(self, alpha, beta, gamma, tx, ty, tz, X, Y, theta, focal_length, plane_height):
        """探测器与热尾流的映射关系
        input:
            alpha, beta, gamma:绕x轴、y轴、z轴旋转的角度,单位：度;
            tx、ty、tz:相对于x轴、y轴、z轴的平移量;
            像平面坐标系的原始坐标:(X,Y)
        out:变换后在绝对坐标系中的新坐标(x,y)"""
        # 坐标转换，考虑飞机的转向
        alpha = alpha * np.pi / 180
        beta = beta * np.pi / 180
        gamma = gamma * np.pi / 180
        R = np.array([[np.cos(beta) * np.cos(gamma), np.cos(beta) * np.sin(gamma), -np.sin(beta), 0],
                      [np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma),
                       np.sin(alpha) * np.sin(beta) * np.sin(gamma)
                       + np.cos(alpha) * np.cos(gamma), np.sin(alpha) * np.cos(beta), 0],
                      [np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma),
                       np.cos(alpha) * np.sin(beta) * np.sin(gamma)
                       - np.sin(alpha) * np.cos(gamma), np.cos(alpha) * np.cos(beta), 0],
                      [0, 0, 0, 1]])
        x, y, z, _ = np.dot(R, np.array([X, Y, 0, 1])) + np.array([tx, ty, tz - focal_length, 1])  # 像平面坐标系转到海洋坐标系

        x0, y0, z0, _ = np.dot(R, np.array([0, 0, focal_length, 1])) + np.array([tx, ty, tz, 1])  # 探测器原点在像平面的坐标转到海洋坐标系

        v1 = np.array([x - x0, y - y0])

        v2 = np.array([1, 0])
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        psi = np.degrees(arccos)  # 0-180度
        psi = psi * np.pi / 180
        if np.isnan(psi):
            psi = 0

        # 映射
        # zt = tz - qz - focal_length
        zt = plane_height - focal_length
        if y >= y0:
            x = x + zt * np.tan(theta) * np.cos(psi)  # 映射到海平面
            y = y + zt * np.tan(theta) * np.sin(psi)
        else:
            x = x + zt * np.tan(theta) * np.cos(psi)
            y = y - zt * np.tan(theta) * np.sin(psi)

        # z = qz

        return x, y

    def cal_area_range(self, vel):
        # self.L = 668 * vel - 136.8  # 热尾流浮升至海面的水平距离,单位m，此时温度变为290.1K
        # self.S1 = 348.53 * vel - 4.6  # 热尾流影响区域的等效特征长度(温度大于290.1K的区域)
        # self.Sw = -51.24 * vel ** 2 + 89.337 * vel + 9.13  # 热尾流影响区域的等效特征宽度

        self.L_range = []
        self.tem_range_max = []
        self.L_range.append(222.7 * vel - 136.8)  # 进入热尾流影响区域
        self.tem_range_max.append(290.1)  # 区域1 ：此时温度变为290.1K

        self.L_range.append(max(self.L_range[-1] + 17.95 * np.power(vel, 1 / 2) - 10.935, 5))  # 到达温度最高点
        self.tem_range_max.append(np.clip(1.76467 * np.exp(-2 * vel) + 290.103, 290.1, 290.3))  # 区域2 ： 快速上升

        self.L_range.append(max(self.L_range[-1] - 503.6863 * np.exp(-0.51 * vel) + 503.883, 5))  # 离开热尾流影响区域
        self.tem_range_max.append(290.1)  # 区域3 ： 缓慢下降

        self.L_range.append(self.L_range[-1] - 418.537 * np.exp(-0.5 * vel) + 440.32)  # 最后画一段意思一下
        self.tem_range_max.append(290.0)

    def cal_area(self, L):
        # vel单位是米每秒
        # 区域1 ： 保持不变，依旧是290K

        if L <= self.L_range[0] // 3:  # 区域1 ： 保持不变，
            return self.sea_temp
        elif L <= self.L_range[0]:
            return self.tem_range_max[0] + (L - self.L_range[0]) * (self.tem_range_max[0] - 290) / (
                    self.L_range[0] - self.L_range[0] // 2) + self.sea_temp - 290
        elif L <= self.L_range[1]:  # 区域2 ： 快速上升
            return self.tem_range_max[0] + (L - self.L_range[0]) * (self.tem_range_max[1] - self.tem_range_max[0]) / (
                    self.L_range[1] - self.L_range[0]) + self.sea_temp - 290
        elif L <= self.L_range[2]:  # 区域3 ： 缓慢下降，直到离开热尾流影响区域
            return self.tem_range_max[1] + (L - self.L_range[1]) * (self.tem_range_max[2] - self.tem_range_max[1]) / (
                    self.L_range[2] - self.L_range[1]) + self.sea_temp - 290
        elif L <= self.L_range[3]:  # 最后画一段意思一下
            return self.tem_range_max[2] + (L - self.L_range[2]) * (self.tem_range_max[3] - self.tem_range_max[2]) / (
                    self.L_range[3] - self.L_range[2]) + self.sea_temp - 290
        else:
            return self.sea_temp

    def cal_temp(self, x, y, qx, qy, outside_render_angle: float = 19.47):
        # 用于计算潜艇附近的热量
        psi = self.sub_psi * np.pi / 180
        outside_render_angle = outside_render_angle * np.pi / 180
        v1 = np.array([-np.sin(psi), -np.cos(psi)])
        v2 = np.array([x - qx, y - qy])
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        temp = self.sea_temp
        if arccos <= outside_render_angle:  # 在范围内
            # 计算水平距离
            # print('arccos', np.degrees(arccos))
            L = np.cos(arccos) * np.linalg.norm(v2)
            # print('L', self.cal_area(L), L)
            # print('self.L_range', self.L_range)
            # print('self.tem_range_max', self.tem_range_max)
            temp = max(self.cal_area(L) - 0.01 * (arccos / outside_render_angle) ** 2, self.sea_temp)
        return temp

    def fishboat_cal_temp(self, x, y, X, Y, fisherboat, outside_render_angle: float = 70, Kelvin_angle: float = 19.47):
        # 用于计算渔船及其尾流的热量
        psi = fisherboat['psi'] * np.pi / 180
        outside_render_angle = outside_render_angle * np.pi / 180
        v1 = np.array([-np.sin(psi), -np.cos(psi)])
        v2 = np.array([x - fisherboat['fx'], y - fisherboat['fy']])
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        temp = self.sea_temp
        # 渔船处热量
        if arccos <= outside_render_angle:  # 在范围内
            L = np.cos(arccos) * np.linalg.norm(v2)
            if L < fisherboat['length'] + self.detect_dis_x:
                temp = self.sea_temp + np.clip(- 0.39 * np.exp(-0.18 * fisherboat['vel']) + 0.42, 0.1, 0.5)

        # 尾流处热量
        Kelvin_angle = Kelvin_angle * np.pi / 180
        if arccos <= Kelvin_angle:  # 在范围内
            L = np.cos(arccos) * np.linalg.norm(v2)
            if L < 300:
                temp = self.sea_temp - np.clip(np.random.random(), 0.2, 0.3)
            elif L < 500:
                temp = self.sea_temp + np.clip(np.random.random(), 0.18, 0.2)
            elif L < 750:
                temp = self.sea_temp - np.clip(np.random.random(), 0.12, 0.15)
            elif L < 1000:
                temp = self.sea_temp + np.clip(np.random.random(), 0.10, 0.12)

        return temp

    def sea_rad(self, lambda_, emissivity, temp):
        c1 = 3.74 * 1e8
        c2 = 1.439 * 1e4
        return emissivity * np.pi * c1 * lambda_ ** -5 / (np.exp(c2 / (lambda_ * temp)) - 1)

    def cal_rad(self, x, y, fx, fy, fz, temp, lambda_, coff):
        """input:
        x，y: 海面坐标
        fx，fy：飞机位置
        temp：x，y位置的海面温度
        lambda_红外波段：3um——5um，and 8um——12um"""
        dis = np.sqrt((x - fx) ** 2 + (y - fy) ** 2 + fz ** 2)
        # 用于计算热尾流的传递到传感器的能量（需要考虑辐射模型）
        emissivity = 0.98  # 发射率
        Transmittance = 0.4858 * np.exp(-1.008 * dis) + 0.514  # 透过率
        value = Transmittance * integ.quad(self.sea_rad, a=lambda_[0], b=lambda_[1], args=(emissivity, temp))[
            0]  # 热尾流辐射

        noise = np.random.normal(0, 0.2) / coff  # 添加高斯噪声

        value = value + noise
        return value

    def fig_plot(self, matrix, window_size=5):
        smoothed_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                min_i = max(0, i - window_size // 2)
                max_i = min(matrix.shape[0], i + window_size // 2 + 1)
                min_j = max(0, j - window_size // 2)
                max_j = min(matrix.shape[1], j + window_size // 2 + 1)
                window = matrix[min_i:max_i, min_j:max_j]
                smoothed_matrix[i, j] = np.mean(window)

        # plt.imshow(smoothed_matrix, cmap="jet")
        # plt.axis('off')
        # plt.xticks([])
        # plt.yticks([])
        # plt.show()
        return smoothed_matrix

    def sub_detect(self, plane_param, sub_param, env_params, fisherboat_params, plt_show=False):
        # 针对单潜艇而言
        imag_unit = np.zeros((self.image_height, self.image_width))  # 成像单元数为100×100
        self.sea_temp = env_params['temp']
        self.sea_state = env_params['sea_state']
        lambda_ = [8, 14]#波段
        # 坐标映射
        qx, qy, sub_height = lla_to_xyz(sub_param['lat'], sub_param['lon'], sub_param['height'])

        self.sub_psi = 360 + sub_param['psi'] if sub_param['psi'] < 0 else sub_param['psi']
        vel = sub_param['vel']
        alpha = plane_param['phi']  # 滚转角 -- x轴旋转的角度
        beta = plane_param['psi']  # 偏航角 -- y轴旋转的角度
        gamma = plane_param['theta']  # 俯仰角 -- z轴旋转的角度
        plane_lat, plane_lon, plane_height = plane_param['lat'], plane_param['lon'], plane_param['height']
        tx, ty, tz = lla_to_xyz(plane_lat, plane_lon, plane_height)  # 用ot
        # map_matrix = [[(0, 0) for _ in range(self.image_height)] for _ in range(self.image_width)]

        X = -self.image_width // 2
        Y = -self.image_height // 2 + 1
        theta = np.arctan(self.proportion * np.sqrt(X ** 2 + Y ** 2) / self.focal_leng)
        x_min, y_min = self.mapping(alpha, beta, gamma, tx, ty, tz, X, Y, theta, self.focal_leng, plane_height)

        X = self.image_width // 2 - 1
        Y = self.image_height // 2
        theta = np.arctan(self.proportion * np.sqrt(X ** 2 + Y ** 2) / self.focal_leng)
        x_max, y_max = self.mapping(alpha, beta, gamma, tx, ty, tz, X, Y, theta, self.focal_leng, plane_height)
        # print('探测距离为{}km'.format(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 1000))
        self.detect_dis_x = (x_max - x_min) / self.image_width  # 每个像素相对于的探测距离x
        self.detect_dis_y = (y_max - y_min) / self.image_height  # 每个像素相对于的探测距离y
        result = []
        sub_height_find = 0

        if qx >= x_min and qx <= x_max and qy >= y_min and qy <= y_max:
            if self.sea_state == 1:
                find_pro = 0.95
                if abs(sub_param['height']) <= 150:
                    sub_height_find = 1
            elif self.sea_state == 2:
                find_pro = 0.85
                if abs(sub_param['height']) <= 120:
                    sub_height_find = 1
            elif self.sea_state == 3:
                find_pro = 0.7
                sub_height_detect_max = np.clip(5 * sub_param['vel'] / 0.5144 + 40, 25, 80)
                if abs(sub_param['height']) <= sub_height_detect_max:
                   sub_height_find = 1
            elif self.sea_state == 4:
                find_pro = 0.4
                sub_height_detect_max = np.clip(5 * sub_param['vel'] / 0.5144, 15, 60)
                if abs(sub_param['height']) <= sub_height_detect_max:
                    sub_height_find = 1
            else:
                find_pro = 0.1
                sub_height_detect_max = np.clip(2.5 * sub_param['vel'] / 0.5144, 8, 35)
                if abs(sub_param['height']) <= sub_height_detect_max:
                    sub_height_find = 1
            if np.random.uniform(0,1) <= find_pro and sub_height_find:
                find_sub = 1  # 可以找到潜艇
                dis = max(222.7 * vel - 136.8 + 17.95 * np.power(vel, 1 / 2) - 10.935, 5)
                g = geod.Direct(lat1=sub_param['lat'], lon1=sub_param['lon'], azi1=sub_param['psi']+180 + np.random.uniform(-10, 10), s12=dis)
                result.append({'type': '潜艇', 'ref_pos':{"lat": g['lat2'], 'lon': g['lon2']},'find': True})
            else:
                find_sub = 0
        else:
            find_sub = 0  # 不能找到潜艇

        # 探测渔船
        fisherboat_param = []
        find_fish = 0
        for fisherboat in fisherboat_params:
            fx, fy, fz = lla_to_xyz(fisherboat['lat'], fisherboat['lon'], 0)
            if fx >= x_min and fx <= x_max and fy >= y_min and fy <= y_max:
                if self.sea_state == 1:
                    find_pro = 0.95
                elif self.sea_state == 2:
                    find_pro = 0.85
                elif self.sea_state == 3:
                    find_pro = 0.75
                elif self.sea_state == 4:
                    find_pro = 0.6
                else:
                    find_pro = 0.5
                if np.random.uniform(0, 1) <= find_pro:
                    find_fish = 1
                    dis = 500 + np.random.uniform(100, 300)
                    g = geod.Direct(lat1=fisherboat['lat'], lon1=fisherboat['lon'], azi1=fisherboat['psi'] + 180 + np.random.uniform(-10, 10), s12=dis)
                    result.append({'type': '渔船','ref_pos': {"lat": g['lat2'], 'lon': g['lon2']}, 'find': True})
                    fisherboat_param.append(
                        {'fx': fx, 'fy': fy, 'fz': fz, 'length': fisherboat['length'], "weight": fisherboat['weight'],
                         "vel": fisherboat['vel'], 'psi': fisherboat['psi'], 'find': find_fish, 'X': None, 'Y': None, 'dis': 1e5,
                         'x': None, 'y': None})


        coff = np.clip(103.853 * np.exp(sub_param['height'] * 0.1) * np.exp(plane_height * 0.001) - 1.257, 1, 96)

        if plt_show:
            if find_sub or find_fish:
                self.cal_area_range(vel)  # 更新潜艇热尾流区域面积
                for X in range(-self.image_width // 2, self.image_width // 2):
                    for Y in range(-self.image_height // 2 + 1, self.image_height // 2 + 1):
                        theta = np.arctan(self.proportion * np.sqrt(X ** 2 + Y ** 2) / self.focal_leng)
                        x, y = self.mapping(alpha, beta, gamma, tx, ty, tz, X, Y, theta, self.focal_leng, plane_height)
                        X_trans = X + self.image_height // 2
                        Y_trans = -(Y - self.image_height // 2)
                            # map_matrix[Y_trans][X_trans] = (x, y)  # 图片当前位置对应的海面坐标
                        if find_sub:
                            temp = self.cal_temp(x, y, qx, qy)
                        else:
                            temp = 0
                        if find_fish:
                            for fisherboat in fisherboat_param:
                                temp2 = self.fishboat_cal_temp(x, y, Y_trans, X_trans, fisherboat)
                                temp = temp + temp2
                            if find_sub:
                                temp = temp - self.sea_temp * len(fisherboat_param)
                            else:
                                temp = temp - self.sea_temp * (len(fisherboat_param)-1)
                        if temp < self.sea_temp + 0.001:  # 探测器灵敏度为
                            temp = self.sea_temp
                        imag_unit[Y_trans][X_trans] = self.cal_rad(x, y, tx, ty, plane_height, temp, lambda_,
                                                                        coff)
            else:
                imag_unit = self.sea_temp + np.random.normal(0, 0.2, (50, 50)) / coff

        return result, imag_unit



    def detect(self, plane_param, sub_params, fisherboat_params, env_params, plt_show=False):  # 输入环境温度
        result = []
        for sub_param in sub_params:
            result_, imag_unit = self.sub_detect(plane_param, sub_param, env_params, fisherboat_params, plt_show=plt_show)
            result.extend(result_)
        if plt_show:
            return (result, self.fig_plot(imag_unit))
        else:
            return (result, None)
        # self.fig_plot(self.imag_unit)
        # return self.imag_unit


if __name__ == '__main__':
    # 输入渔船、飞机位置、潜艇位置、当前温度
    infrared = infrared()
    lambda_ = [8, 12]  # 红外波段比如3um——5um，and 8um——12um
    plane_param = {'lat': 17.91944, 'lon': 110.48083, 'height': 500, 'phi': 0, 'psi': 0,
                   'theta': 0}  # 偏航角,朝着正北为0度，-180 - 180度
    sub_params = [{'lat': 17.91944, 'lon': 110.48083, 'height': -30, 'vel': 10 * 0.5144, 'psi': -90}]  # 输入速度单位m/s
    fisherboat_params = [
        {'lat': 17.915, 'lon': 110.5, 'length': 5.8, "weight": 2, "vel": 12 * 0.5144,
         'psi': 0},{'lat': 18.0, 'lon': 110.5, 'length': 5.8, "weight": 2, "vel": 12 * 0.5144,
         'psi': 0}]  # 渔船船长和宽度的单位为m， 输入速度单位m/s
    env_temp = 290  # 海水温度
    env_params = {'temp': env_temp, 'sea_state':1} #环境信息
    start = time.time()
    a = infrared.detect(plane_param, sub_params, fisherboat_params, env_params, plt_show=False)
    print(a)
    print('运行时间：', time.time() - start)
