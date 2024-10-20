"""
工具类
pyproj官方文档: https://pyproj4.github.io/pyproj/stable/api/transformer.html
"""
from geographiclib.geodesic import Geodesic
import pyproj
from pyproj import Transformer

geod = Geodesic.WGS84

"""
xyz转经纬度
"""

import numpy as np
import math


class CoordinateTransformer:
    def __init__(self) -> None:
        self.a = 6378245  # 克拉索夫斯基参考椭球的长半轴
        self.b = 6356863  # 短半轴　
        self.C = self.a * self.a * 6366699 / self.b
        # self.e2 = (self.a ** 2 - self.b ** 2) / (self.a ** 2) 
        self.f = (self.a - self.b) / self.a
        self.e2 = self.f * (2 - self.f)
        self.e22 = self.e2 / (1 - self.e2)  # 常熟

    def lla_to_xyz(self, lat, lon, alt):
        transprojr = Transformer.from_crs(
            "EPSG:4326",
            {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
            always_xy=True)
        x, y, z = transprojr.transform(lon, lat, alt, radians=False)
        return x, y, z

    def xyz_to_lla(self, x, y, z):
        transproj = Transformer.from_crs(
            {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
            "EPSG:4326",
            always_xy=True,
        )
        lon, lat, alt = transproj.transform(x, y, z, radians=False)
        return lon, lat, alt

        # # 计算法
    # def lla_to_xyz(self, lat, lon, alt):# ecef 
    #     N = self.a / (1 - self.f * (2-self.f) * math.sin(math.radians(lat)) ** 2) 
    #     x = (N + alt) * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    #     y = (N + alt) * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    #     z = (N * (1 - self.f)**2 + alt) * math.sin(math.radians(lat))
    #     return x, y, z

    # 计算法
    # def xyz_to_lla(self, x, y, z):
    #     lon = math.degrees(math.atan2(y, x))
    #     p = math.sqrt(x**2 + y**2)
    #     theta = math.degrees(math.atan2(z * t.a, p * t.b))
    #     lat = math.degrees(math.atan2(z + t.e22 * t.b * math.sin(math.radians(theta)) ** 3, p - t.e2 * t.a * math.cos(math.radians(theta)) ** 3))


# 平滑转弯
def smooth_fold(cur_angle, tar_angle, d_a):
    if abs(cur_angle - tar_angle) > 1:
        if (90 <= cur_angle and -180 <= tar_angle <= -90) or (
                0 <= cur_angle and -180 <= tar_angle <= -90 and 180 - abs(cur_angle) + 180 - abs(
            tar_angle) < 180) or (
                90 <= cur_angle and -90 <= tar_angle <= 0 and 180 - abs(cur_angle) + 180 - abs(
            tar_angle) < 180):
            if cur_angle >= 180:
                cur_angle -= 360
            if cur_angle > 0:
                cur_angle += d_a
            elif cur_angle < 0:
                cur_angle -= d_a
        elif (cur_angle <= -90 and 90 <= tar_angle <= 180) or (
                0 <= tar_angle <= 90 and cur_angle <= -90 and 180 - abs(cur_angle) + 180 - abs(
            tar_angle) < 180) or (
                90 <= tar_angle <= 180 and cur_angle <= 0 and 180 - abs(cur_angle) + 180 - abs(
            tar_angle) < 180):
            if cur_angle <= -180:
                cur_angle += 360
            cur_angle -= d_a
        else:
            if cur_angle > tar_angle:
                cur_angle -= d_a
            else:
                cur_angle += d_a
        return cur_angle
    else:
        return tar_angle


if __name__ == '__main__':
    t = CoordinateTransformer()
    lat, lon, alt2 = 24, 120, 3000
    x, y, z1 = t.lla_to_xyz(lat=lat, lon=lon, alt=alt2)

    print(x, y, z1)

    lat, lon, alt1 = 24.1, 120, 3000
    x, y, z2 = t.lla_to_xyz(lat=lat, lon=lon, alt=alt1)
    print(z2 - z1)

    print(3000 * math.sin(math.radians(24.1)) - 3000 * math.sin(math.radians(24)))
    # print((alt1 - alt2) * math.sin(math.radians(lat)))
    # print(z1 -z2)
    # # print(x, y, z)
    # lon, lat, alt = t.xyz_to_lla(x, y ,z)
    # print(lon, lat, alt)
