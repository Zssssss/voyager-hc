"""
态势解析类
"""
"""
态势类
"""
import argparse
import base64
from io import BytesIO
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84
import random


######################## 环境态势 ########################
class EnvGlobalObservation:
    """
    环境
    """

    def __init__(self, args):
        # self.args = args

        # 渔船
        if args.Env_params_Inf_num == '1':  # 较少干扰物
            self.fishing_boat_nums = np.random.randint(0, 2)  # 渔船数量
            self.cargo_ship_nums = np.random.randint(0, 2)  # 货轮数量
            self.vortex_nums = np.random.randint(0, 2)  # 旋涡数量
            self.wreck_nums = np.random.randint(0, 2)  # 沉船数量
            self.fish_nums = np.random.randint(0, 2)  # 鱼群数量

        elif args.Env_params_Inf_num == '2':  # 适中干扰物
            self.fishing_boat_nums = np.random.randint(2, 4)
            self.cargo_ship_nums = np.random.randint(2, 4)
            self.vortex_nums = np.random.randint(2, 4)
            self.wreck_nums = np.random.randint(2, 4)
            self.fish_nums = np.random.randint(2, 4)

        elif args.Env_params_Inf_num == '3':  # 较多干扰物
            self.fishing_boat_nums = np.random.randint(4, 6)
            self.cargo_ship_nums = np.random.randint(4, 6)
            self.vortex_nums = np.random.randint(4, 6)
            self.wreck_nums = np.random.randint(4, 6)
            self.fish_nums = np.random.randint(4, 6)

        else:
            raise ValueError('干扰物平均数量导入错误')

        self.fishing_boats = [Fishing_Boat(args) for _ in range(self.fishing_boat_nums)]
        self.cargo_ships = [Cargo_Ship(args) for _ in range(self.cargo_ship_nums)]
        self.vortexs = [Vortex(args) for _ in range(self.vortex_nums)]
        self.wrecks = [Wreck(args) for _ in range(self.wreck_nums)]
        self.fishs = [Fish(args) for _ in range(self.fish_nums)]

######################## 渔船 ########################
class Fishing_Boat:
    """
    渔船位置初始化
    小型渔船：长<12m，宽<2.7m
    中型渔船：12m<=长<=24m，2.7<=宽<=6m
    大型渔船：长>24m，宽>6m
    """

    def __init__(self, args):
        super(Fishing_Boat, self).__init__()
        # 初始化位置
        self.lon = random.uniform(args.blue_sub_field["min_lon"] - 0.4, args.blue_sub_field["max_lon"] + 0.4)
        self.lat = random.uniform(args.blue_sub_field["min_lat"] - 0.4, args.blue_sub_field["max_lat"] + 0.4)
        self.vel = 0
        self.angle = random.uniform(-180, 180)
        kind = random.randint(0, 2)
        self.init_size(kind)
    def init_size(self, choose=0):
        if choose == 0:
            k = np.random.uniform(0.42, 1)
            self.length = k * 12
            self.width = np.clip(k * 2.7, 1.5, 2.7)
        elif choose == 1:
            k = np.random.uniform(0.5, 1)
            self.length = k * 24
            self.width = k *6.6-0.6
        else:
            k = np.random.uniform(1, 1.5)
            self.length = k * 24
            self.width = k * 6


######################## 货船 ########################
class Cargo_Ship:
    """
    货轮位置初始化
    """

    def __init__(self, args):
        super(Cargo_Ship, self).__init__()
        # 初始化位置
        self.lon = random.uniform(args.blue_sub_field["min_lon"] - 0.4, args.blue_sub_field["max_lon"] + 0.4)
        self.lat = random.uniform(args.blue_sub_field["min_lat"] - 0.4, args.blue_sub_field["max_lat"] + 0.4)
        self.angle = random.uniform(-180, 180)

######################## 漩涡 ########################
class Vortex:
    """
        旋涡位置初始化
        """

    def __init__(self, args):
        super(Vortex, self).__init__()
        # 初始化位置
        self.lon = random.uniform(args.blue_sub_field["min_lon"] - 0.4, args.blue_sub_field["max_lon"] + 0.4)
        self.lat = random.uniform(args.blue_sub_field["min_lat"] - 0.4, args.blue_sub_field["max_lat"] + 0.4)

######################## 沉船 ########################
class Wreck:
    """
        沉船位置初始化
        """

    def __init__(self, args):
        super(Wreck, self).__init__()
        # 初始化位置
        self.lon = random.uniform(args.blue_sub_field["min_lon"] - 0.4, args.blue_sub_field["max_lon"] + 0.4)
        self.lat = random.uniform(args.blue_sub_field["min_lat"] - 0.4, args.blue_sub_field["max_lat"] + 0.4)

######################## 鱼群 ########################
class Fish:
    """
        鱼群位置初始化
        """
    def __init__(self, args):
        super(Fish, self).__init__()
        # 初始化位置
        self.lon = random.uniform(args.blue_sub_field["min_lon"] - 0.4, args.blue_sub_field["max_lon"] + 0.4)
        self.lat = random.uniform(args.blue_sub_field["min_lat"] - 0.4, args.blue_sub_field["max_lat"] + 0.4)
        self.height = random.uniform(-150, -30)
