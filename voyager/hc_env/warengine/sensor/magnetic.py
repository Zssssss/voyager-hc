# -*- coding: utf-8 -*-
"""
@Time : 2024/5/22 14:32
@Auth : Lin Yang
@File : time_mag.py
@IDE : PyCharm
"""
import math
import random
import time

import numpy as np
from geographiclib.geodesic import Geodesic
from pyproj import CRS
from pyproj import Transformer
import matplotlib.pyplot as plt

geod = Geodesic.WGS84

crs_WGS84 = CRS.from_epsg(4326)
crs_WebMercator = CRS.from_epsg(3857)  # Web墨卡托投影坐标系


def lla_to_xyz(lat, lon, alt):
    transprojr = Transformer.from_crs(
        "EPSG:4326",
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, always_xy=True)
    x, y, z = transprojr.transform(lon, lat, alt, radians=False)

    return x, y, z


def xyz_to_lla(x, y, z):
    transproj = Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}, "EPSG:4326", always_xy=True)
    lon, lat, alt = transproj.transform(x, y, z, radians=False)

    return lon, lat, alt


def draw_figure(data=None, tag=False):
    # 绘制图表
    if tag:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.title("磁异常检测特征图")
        plt.plot(data)
        plt.xlabel("t / s")
        plt.ylabel("intensity / Nt")
        plt.show()


class MagneticObject:
    def __init__(self, lat, lon, alt, M_m):
        self.depth = alt
        self.lat, self.lon, self.alt = lla_to_xyz(lat, lon, alt)

        # 磁性物体的磁矩大小
        self.M_m = M_m


class Magnetic:
    def __init__(self, lat, lon, alt, R_m):
        self.depth = alt
        self.lat, self.lon, self.alt = lla_to_xyz(lat, lon, alt)
        # 探测长度
        self.R_m = R_m

    def cover_area(self, H, h):
        """
        R_m : 表示探测长度
        H : 表示观测点高度
        h : 表示磁性物体下潜深度
        """
        # h : sub航行深度
        # assert H - h <= self.R_m, "h + H 的和不能大于探测长度 R_m"
        if H - h > self.R_m:
            return 0
        # 无人机水下探测半径
        r = int(math.sqrt(self.R_m ** 2 - (H - h) ** 2))
        return r


class Environment:

    def __init__(self, objects: list, magnetic: Magnetic, number: int):
        self.objects = objects
        self.magnetic = magnetic
        self.number = number
        self.threshold = 1e-08

    @classmethod
    def generate_objects(cls, number: int):
        """
        生成磁性物体
        """
        random.seed = 42
        random_objects = []
        # 生成磁性物体
        for i in range(number):
            lat = random.uniform(20, 34.5)
            lon = random.uniform(118, 133)
            alt = random.uniform(-100, 0)
            R_m = random.uniform(40, 80)
            magnetic_object = MagneticObject(lat, lon, alt, R_m)
            random_objects.append(magnetic_object)
        return random_objects

    # 随机添加坐标噪声值
    @classmethod
    def add_gaussian_noise(cls, lat, lon, alt, mean, std_dev):
        noise = random.gauss(mean, std_dev)  # 生成高斯噪声值
        lat_noisy_number = lat + noise
        lon_noisy_number = lon + noise
        alt_noisy_number = alt + noise
        lon, lat, alt = xyz_to_lla(lat_noisy_number, lon_noisy_number, alt_noisy_number)
        return lon, lat, alt

    def magnetic_intensity(self, object: MagneticObject):
        distance = math.sqrt((object.lat - self.magnetic.lat) ** 2 + (object.lon - self.magnetic.lon) ** 2)
        sin_value = (abs(object.lat - self.magnetic.lat)) / distance

        # 计算当前磁性物体在观测点的磁场强度值
        value = object.M_m / (distance ** 3) * math.sqrt(1 + 3 * (sin_value ** 2))
        return value

    def detect_intensity(self):
        detect_object = []
        sum_value = 0
        magnetic_objects = self.generate_objects(self.number)
        self.objects.extend(magnetic_objects)
        for object in self.objects:

            # 计算当前探测视野半径
            r = self.magnetic.cover_area(H=self.magnetic.depth, h=object.depth)

            # 如果水下磁性物体深度多于 200m，则当前磁性物体探测结果为 0
            if object.depth <= -200:
                continue

            # 判断当前磁性物体是否在磁探仪的探测范围内，不在则当前磁性物体探测结果为 0
            if (object.lat - self.magnetic.lat) ** 2 + (object.lon - self.magnetic.lon) ** 2 > r ** 2:
                continue

            # 计算当前探测视野半径内物体数量
            value = self.magnetic_intensity(object)
            if value <= self.threshold:
                continue
            sum_value += value

            # 三维空间坐标
            lon, lat, alt = self.add_gaussian_noise(object.lat, object.lon, object.alt, 1, 3)
            detect_object.append((lat, lon, alt))

        return sum_value, detect_object


if __name__ == '__main__':
    start = time.time()
    submarine = MagneticObject(lat=22.1000, lon=120.111, alt=-100, M_m=80)
    freighter = MagneticObject(lat=22.1, lon=120.111113, alt=0, M_m=100)
    fishing_boat = MagneticObject(lat=22.12, lon=120.12, alt=-50, M_m=100)

    magnetic = Magnetic(lat=22.1, lon=120.11110, alt=200, R_m=800)

    objects_list = [submarine, freighter, fishing_boat]

    env = Environment(objects_list, magnetic=magnetic, number=0)
    seq_detect_intensity = []
    seq_detect_pos = []
    sign = True
    n = 0
    # while sign:
    #     detect_value = env.detect_intensity()
    #     seq_detect_intensity.append(detect_value)
    #     if n == 100:
    #         sign = False
    # for i in range(10):
    #     detect_value, detect_object_pos = env.detect_intensity()
    #     seq_detect_intensity.append(detect_value)
    #     seq_detect_pos.append(detect_object_pos)
    detect_value, detect_object_pos = env.detect_intensity()
    seq_detect_intensity.append(detect_value)
    seq_detect_pos.append(detect_object_pos)
    print(f"seq_detect_intensity: {seq_detect_intensity}")
    print(f"seq_detect_pos: {seq_detect_pos}")
    draw_figure(seq_detect_intensity, tag=True)
    end = time.time()
    print(end - start)
