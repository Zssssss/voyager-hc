"""
# python
# -*- coding:utf-8 -*-
@Project : SQ_ChenWei
@File : fishing_boat.py
@Author : 一杯可乐
@Time : 2024/3/18 17:20
@Description : 渔船动力学模型
"""

import random
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84


class FishingBoatControl:
    def __init__(self, fishing_boat):
        self.lon, self.lat = fishing_boat.lon, fishing_boat.lat
        self.last_lat = self.lat
        self.last_lon = self.lon

        # 设置初始速度和目标速度
        self.v_j = random.randint(9, 12)
        self.v = self.v_j * 1.852 * 1000 / 3600  # 速度-m/s

        # 运动方向：相对经度偏离角度
        cur_d = random.uniform(-90, 90)
        self.d = round(cur_d, 2)

        # 渔船运动模式
        self.move_mode = random.randint(0, 2)
        self.move_mode_1_i = 0  # 记录模式1的时间
        self.move_mode_2_i = 0  # 记录模式2的时间

    def fishing_boat_move(self):
        """
        渔船移动设定
        一般渔船的时速约为9到12节，也就是大约18至22公里时速。
        :return:
        """
        self.last_lat = self.lat
        self.last_lon = self.lon

        s12 = 0

        # 直线行驶，并且速度和角度有一定随机偏移
        if self.move_mode == 0:
            # 当前速度
            self.v = self.v + random.uniform(-0.5, 0.5)
            # 计算前进距离
            s12 = self.v * 1
            # 当前方向
            self.d = self.d + random.uniform(-1, 1)
            # # 计算当前经纬度位置
            # cur_pos = geod.Direct(self.last_lat, self.last_lon, cd, s12)
            # self.lon = cur_pos['lon2']
            # self.lat = cur_pos['lat2']
        # 捕鱼：先放网，速度慢，再收网，速度快
        elif self.move_mode == 1:
            # 低速前进
            if self.move_mode_1_i < 200:
                self.v = self.v + random.uniform(-0.005, 0.005)
                s12 = self.v * 1
            # 加速
            elif 200 <= self.move_mode_1_i < 300:
                self.v = self.v + 0.02 + random.uniform(-0.0001, 0.0001)
                s12 = self.v * 1
            # 保持高速
            elif 300 <= self.move_mode_1_i < 500:
                self.v = self.v + random.uniform(-0.0001, 0.0001)
                s12 = self.v * 1
            # 减速
            elif 500 <= self.move_mode_1_i < 700:
                self.v = self.v - 0.02 + random.uniform(-0.0001, 0.0001)
                s12 = self.v * 1
            # 归零
            elif self.move_mode_1_i >= 700:
                self.v = self.v + random.uniform(-0.0001, 0.0001)
                s12 = self.v * 1
                self.move_mode_1_i = 0
            # 当前方向
            self.d = self.d + random.uniform(-1, 1)

            self.move_mode_1_i += 1
        # 完成捕鱼作业，掉头
        elif self.move_mode == 2:
            # 低速前进
            if self.move_mode_2_i < 200:
                self.v = self.v + random.uniform(-0.005, 0.005)
                s12 = self.v * 1
            # 减速掉头
            elif 200 <= self.move_mode_2_i < 350:
                self.v = self.v - 0.02 + random.uniform(-0.0001, 0.0001)
                s12 = self.v * 1
                self.d = self.d + 1
            # 加速
            elif 350 <= self.move_mode_2_i < 500:
                self.v = self.v + 0.02 + random.uniform(-0.0001, 0.0001)
                s12 = self.v * 1
            # 直线行驶
            else:
                self.v = self.v + random.uniform(-0.0001, 0.0001)
                s12 = self.v * 1
                self.d = self.d + random.uniform(-0.1, 0.1)
            self.move_mode_2_i += 1
        # 计算当前经纬度位置
        cur_pos = geod.Direct(self.last_lat, self.last_lon, self.d, s12)
        self.lon = cur_pos['lon2']
        self.lat = cur_pos['lat2']
        # print("fishing_boat_move_pos:")
        # print(self.lon, self.lat)
        return self.lon, self.lat, self.d, s12/0.5144
