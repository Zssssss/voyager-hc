"""
# python
# -*- coding:utf-8 -*-
@Project : SQ_ChenWei
@File : cargo_ship.py
@Author : 一杯可乐
@Time : 2024/3/19 9:02
@Description : 货轮动力学模型
"""

import random
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84


class CargoShipControl:
    """
    货轮移动
    货轮速度：15~20节
    """

    def __init__(self, cargo_ship):
        self.lon, self.lat = cargo_ship.lon, cargo_ship.lat

        self.last_lat = self.lat
        self.last_lon = self.lon

        # 设置初始速度和目标速度
        self.v_j = random.randint(15, 20)
        self.v = self.v_j * 1.852 * 1000 / 3600  # 速度-m/s

        # 运动方向：相对经度偏离角度
        cur_d = random.uniform(-90, 90)
        self.d = round(cur_d, 2)

    def cargo_ship_move(self):
        self.last_lat = self.lat
        self.last_lon = self.lon

        # 直线行驶，并且速度和角度有一定随机偏移
        # 当前速度
        self.v = self.v + random.uniform(-0.5, 0.5)
        # 计算前进距离
        s12 = self.v * 1
        # 当前方向
        self.d = self.d + random.uniform(-1, 1)

        # 计算当前经纬度位置
        cur_pos = geod.Direct(self.last_lat, self.last_lon, self.d, s12)
        self.lon = cur_pos['lon2']
        self.lat = cur_pos['lat2']
        return self.lon, self.lat, self.d
