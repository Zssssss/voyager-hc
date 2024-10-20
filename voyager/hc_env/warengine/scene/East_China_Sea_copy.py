"""面向东海SQ，形成场景文件"""
# -*- coding: utf-8 -*-

"""
多智能体搜潜
"""
import base64
import random
from io import BytesIO

import matplotlib.pyplot as plt
import argparse
import asyncio
import websockets
import json
import numpy as np
from geographiclib.geodesic import Geodesic

from cmds.PlaneAgent import PlaneDecision
from cmds.SubAgent import SubDecision
from warengine.commands.plane_command import CmdType, RedObjType
from warengine.commands.sub_command import SubmarineCommand, BlueObjType

# from cmds.PlaneAgent import PlaneDecision
from cmds.SubAgent import SubDecision
from agents.red_agent import RedAgent
from warengine.commands.plane_command import CmdType
from warengine.commands.sub_command import BlueObjType
from warengine.entity.plane_control import PlaneControl
from warengine.obs.blue_obs import BlueGlobalObservation, Jammer, acoustic_bait
from warengine.obs.env_obs import EnvGlobalObservation
from warengine.obs.red_obs import RedGlobalObservation, Buoy, passvive_sonar_combination
from geographiclib.geodesic import Geodesic
import time
# from SQ.model.plane_control import PlaneControl
from warengine.entity.sub_control import SubControl
from warengine.entity.usv_control import UsvControl
import math
from warengine.entity.cargo_ship import CargoShipControl
from warengine.entity.fishing_boat import FishingBoatControl
import multiprocessing

geod = Geodesic.WGS84


class Result:
    BLUE_WIN = 1  # 潜艇赢
    GO_ON = 0  # 继续
    RED_WIN = -1  # 飞机赢


class StartData:
    def __init__(self):
        self.multi_sensor_img = None
        plt.clf()
        self.start = True


class ReportInfo:
    def __init__(self, simtime, report_lat, report_lon, report_course, report_vel, sub_lat, sub_lon, sub_course,
                 sub_vel):
        self.simtime = simtime
        self.report_lat = report_lat
        self.report_lon = report_lon
        self.report_course = report_course
        self.report_vel = report_vel

        self.sub_lat = sub_lat
        self.sub_lon = sub_lon
        self.sub_course = sub_course
        self.sub_vel = sub_vel


class SouthCall:
    def __init__(self, args, task_id, episode_i):
        self.args = args
        self.mode = [0 for _ in range(args.submarine_nums)]  # 0 -- 检查SQ， 1--应召SQ
        self.uav_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in range(self.args.uav_nums)]  # 初始化
        self.usv_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in range(self.args.usv_nums)]  # 初始化
        self.sub_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in range(self.args.submarine_nums)]  # 初始化
        self.plane_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in range(self.args.plane_nums)]  # 运九初始化
        self.red_frigate_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in
                                     range(self.args.red_frigate_nums)]  # 护卫舰初始化
        self.red_maritime_ship_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in
                                           range(self.args.red_maritime_ship_nums)]  # 海警船初始化
        self.red_J20_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in
                                 range(self.args.red_J20_nums)]  # 歼20飞机初始化,位于机场
        self.red_H6_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in range(self.args.red_H6_nums)]  # 轰六飞机初始化,位于机场
        self.red_Elec_plane_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in
                                        range(self.args.red_Elec_plane_nums)]  # 电子战飞机初始化,位于机场
        self.red_sub_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in range(self.args.red_sub_nums)]  # 红方潜艇初始化

        # self.sub_target_pos = [{"lat": 0, "lon": 0} for _ in range(self.args.submarine_nums)]  # 初始化
        self.blue_patrol_ship_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in
                                          range(self.args.blue_patrol_ship_nums)]  # 蓝方巡逻舰初始化
        self.blue_destroyer_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in
                                        range(self.args.blue_destroyer_nums)]  # 蓝方巡逻舰初始化
        self.blue_P1_plane_init_pos = [{"lat": 0, "lon": 0, 'height': 0} for _ in
                                       range(self.args.blue_P1_plane_nums)]  # 蓝方P1飞机初始化
        self.blue_F15_plane_init_pos = [
            {"lat": self.args.blue_airport['lat'], "lon": self.args.blue_airport['lon'], 'height': 0} for _ in
            range(self.args.blue_F15_plane_nums)]  # 蓝方F15飞机初始化,位于机场
        self.blue_Elec_plane_init_pos = [
            {"lat": self.args.blue_airport['lat'], "lon": self.args.blue_airport['lon'], 'height': 0} for _ in
            range(self.args.blue_Elec_plane_nums)]  # 蓝方电子战飞机初始化,位于机场

        self.battle_field = args.battle_field
        self.seach_field = {**{"center": {"lat": 0, "lon": 0}},
                            **{"size": {"width": args.search_area_width, "height": args.search_area_height}}, **{
                f"uav{i}": {"width": args.uav_area_width, "height": args.uav_area_height} for i in
                range(self.args.uav_nums)}, **{
                f"usv{i}": {"width": args.uav_area_width, "height": args.usv_area_height} for i in
                range(self.args.usv_nums)}}
        self.args.blue_sub_field = {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}  # 蓝方潜艇的战斗区域

        # 仿真结果统计
        self.result_statistics = {}
        # 蓝方仿真结果
        self.sub_type = args.sub_name
        self.sub_pre_pos = [[0, 0] for _ in range(args.submarine_nums)]  # 潜艇上一时刻位置
        self.sub_low_speed_sailing_duration = [0 for _ in range(args.submarine_nums)]  # 低速航行时长(速度小于2节)，小时
        self.sub_routine_speed_sailing_duration = [0 for _ in range(args.submarine_nums)]  # 常规航行时长(速度6-8节)，小时
        self.sub_high_speed_sailing_duration = [0 for _ in range(args.submarine_nums)]  # 常规航行时长(速度6-8节)，小时
        self.sub_total_navigation_mileage = [0 for _ in range(args.submarine_nums)]  # 航行总里程，米
        self.sub_initial_exposure_time = 0  # 初始暴露时间
        self.sub_bait_deployed_num = 0  # 声诱饵投放数量
        self.sub_jammer_deployed_num = 0  # 干扰器投放数量
        self.sub_velocity_list = [[] for _ in range(args.submarine_nums)]  # 潜艇速度记录
        self.sub_action_list = [[] for _ in range(args.submarine_nums)]  # 潜艇动作记录
        self.sub_state_list = [[] for _ in range(args.submarine_nums)]  # 潜艇状态记录
        # 红方仿真结果
        # 无人机
        self.uav_type = args.uav_name
        self.uav_pre_pos = [[0, 0] for _ in range(args.uav_nums)]  # 无人艇上一时刻位置
        self.uav_total_duration_call_point = [0 for _ in range(args.uav_nums)]  # 到达应召点总时长，小时
        self.uav_time_first_identified_sub = 0  # 第一次识别到潜艇花费时长，小时
        self.uav_total_navigation_mileage = [0 for _ in range(args.uav_nums)]  # 航行总里程，米
        self.uav_sonar_buoy_num = {"passive": 0, "activate": 0}  # 耗费声呐浮标数量
        self.uav_passive_sonar_survival_rate = 0  # 被动声呐存活率
        self.uav_active_sonar_survival_rate = 0  # 主动声呐存活率
        self.uav_target_recognition_accuracy = 0  # 目标识别准确率
        # 无人艇
        self.usv_type = args.usv_name
        self.usv_pre_pos = [[0, 0] for _ in range(args.usv_nums)]  # 无人艇上一时刻位置
        self.usv_total_navigation_mileage = [0 for _ in range(args.usv_nums)]  # 航行总里程，米
        self.usv_total_duration_call_point = 0  # 到达应召点总时长，小时
        self.usv_time_first_identified_sub = 0  # 第一次识别到潜艇花费时长，小时
        self.sonar_update_times = args.sonar_update_times
        self.touch_buoys = []
        self.blue_sub_info = [[] for _ in range(self.args.submarine_nums)]  # 蓝方潜艇角度信息

    def pos_change(self, center_lat, center_lon, width, height):
        """将任务海区的经纬度，从中心位置转移到左上角"""
        h1 = width / 2
        h2 = height / 2
        course = -math.degrees(math.atan(h1 / h2))  # 转化为角度
        d = np.sqrt(h1 ** 2 + h2 ** 2)
        g = geod.Direct(lat1=center_lat, lon1=center_lon, azi1=course, s12=d)
        return g['lat2'], g['lon2']

    def get_init_pos(self):
        """根据想定初始化所有智能体的位置"""
        # 先确定无人艇、无人机任务海区的中间位置 --- 位于黄尾屿以东，中心距离该岛45海里
        course = np.random.uniform(80, 100)
        d = np.random.uniform(10 * 1.852 * 1000,
                              12 * 1.852 * 1000)  # 距离修改过 d = np.random.uniform(30*1.852*1000, 32*1.852*1000)
        g = geod.Direct(lat1=self.args.Huangwei['lat'], lon1=self.args.Huangwei['lon'], azi1=course, s12=d)
        self.seach_field['center']['lat'] = g['lat2']
        self.seach_field['center']['lon'] = g['lon2']

        # 红方态势初始化
        for id in range(self.args.uav_nums):
            if id == 0:
                h1 = self.seach_field['uav0']['width'] / 2
                h2 = self.seach_field['usv1']['height'] + self.seach_field['uav0']['height'] / 2
                course = -math.degrees(math.atan(h1 / h2))  # 转化为角度
            elif id == 1:
                h1 = self.seach_field['uav1']['width'] / 2
                h2 = self.seach_field['uav1']['height'] / 2
                course = math.degrees(math.atan(h1 / h2))
            elif id == 2:
                h1 = self.seach_field['uav2']['height'] / 2
                h2 = self.seach_field['uav2']['width'] / 2
                course = -(math.degrees(math.atan(h1 / h2)) + 90)
            else:
                h1 = self.seach_field['uav3']['height'] / 2 + self.seach_field['usv2']['height']
                h2 = self.seach_field['uav3']['width'] / 2
                course = math.degrees(math.atan(h1 / h2)) + 90

            d = np.sqrt(h1 ** 2 + h2 ** 2)
            g = geod.Direct(lat1=self.seach_field['center']['lat'], lon1=self.seach_field['center']['lon'], azi1=course,
                            s12=d)
            self.uav_init_pos[id]['lat'], self.uav_init_pos[id]['lon'] = self.pos_change(
                g['lat2'], g['lon2'], self.seach_field['uav{}'.format(id)]['width'],
                self.seach_field['uav{}'.format(id)]['height'])  # 将无人机初始位置确定在独立海域的左上角
            self.uav_init_pos[id]['height'] = self.args.uav_height

        for id in range(self.args.usv_nums):
            if id == 0:
                h1 = self.seach_field['usv0']['width'] / 2
                h2 = self.seach_field['uav1']['height'] + self.seach_field['usv0']['height'] / 2
                course = math.degrees(math.atan(h1 / h2))
            elif id == 1:
                h1 = self.seach_field['usv1']['width'] / 2
                h2 = self.seach_field['usv1']['height'] / 2
                course = -math.degrees(math.atan(h1 / h2))
            elif id == 2:
                h1 = self.seach_field['usv2']['height'] / 2
                h2 = self.seach_field['usv2']['width'] / 2
                course = math.degrees(math.atan(h1 / h2)) + 90
            else:
                h1 = self.seach_field['usv3']['height'] / 2 + self.seach_field['uav2']['height']
                h2 = self.seach_field['usv3']['width'] / 2
                course = -(math.degrees(math.atan(h1 / h2)) + 90)
            d = np.sqrt(h1 ** 2 + h2 ** 2)
            g = geod.Direct(lat1=self.seach_field['center']['lat'], lon1=self.seach_field['center']['lon'], azi1=course,
                            s12=d)
            self.usv_init_pos[id]['lat'], self.usv_init_pos[id]['lon'] = self.pos_change(
                g['lat2'], g['lon2'], self.seach_field['usv{}'.format(id)]['width'],
                self.seach_field['usv{}'.format(id)]['height'])  # 将无人艇任务海域的初始位置从中心位置转移到左上角

        # 大鲸级潜艇海域中心黄尾屿正西50海里，距离修改过
        g = geod.Direct(lat1=self.args.Huangwei['lat'], lon1=self.args.Huangwei['lon'], azi1=-90,
                        s12=10 * 1.852 * 1000)  # s12=38 * 1.852 * 1000
        centor_pos = {"lat": g['lat2'], "lon": g['lon2']}
        g = geod.Direct(lat1=centor_pos['lat'], lon1=centor_pos['lon'], azi1=-90,
                        s12=28 * 1.852 * 1000)  # 正方形海域，边长60海里，正常应该是s12=30 * 1.852 * 1000，距离修改过
        self.args.blue_sub_field['min_lon'] = g['lon2']
        g = geod.Direct(lat1=centor_pos['lat'], lon1=centor_pos['lon'], azi1=0, s12=28 * 1.852 * 1000)
        self.args.blue_sub_field['max_lat'] = g['lat2']
        g = geod.Direct(lat1=centor_pos['lat'], lon1=centor_pos['lon'], azi1=180, s12=28 * 1.852 * 1000)
        self.args.blue_sub_field['min_lat'] = g['lat2']
        g = geod.Direct(lat1=centor_pos['lat'], lon1=centor_pos['lon'], azi1=90, s12=28 * 1.852 * 1000)
        self.args.blue_sub_field['max_lon'] = g['lon2']

        # 确定蓝方潜艇的位置
        for id in range(self.args.submarine_nums):
            if id == 0:  # 大鲸级潜艇，海域中心黄尾屿正西50海里，正方形，边长60海里
                course = np.random.uniform(110, 160)
                d = np.random.uniform(10 * 1.852 * 1000, 12 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.blue_sub_field['max_lat'], lon1=self.args.blue_sub_field['min_lon'],
                                azi1=course, s12=d)
                self.sub_init_pos[id]['lat'], self.sub_init_pos[id]['lon'] = g['lat2'], g['lon2']
                self.sub_init_pos[id]['height'] = self.args.sub_height

            if id == 1:  # 苍龙级潜艇，搜索钓鱼岛西南海域60海里，海域半径30海里
                course = np.random.uniform(-110, -170)
                d = np.random.uniform(58 * 1.852 * 1000, 62 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.Diaoyu['lat'], lon1=self.args.Diaoyu['lon'], azi1=course, s12=d)
                centor_pos = {"lat": g['lat2'], "lon": g['lon2']}
                course = np.random.uniform(-35, -55)
                d = np.random.uniform(40 * 1.852 * 1000, 42 * 1.852 * 1000)
                g = geod.Direct(lat1=centor_pos['lat'], lon1=centor_pos['lon'], azi1=course, s12=d)
                self.sub_init_pos[id]['lat'], self.sub_init_pos[id]['lon'] = g['lat2'], g['lon2']

                # course = np.random.uniform(0, 360)
                # d = np.random.uniform(28 * 1.852 * 1000, 32 * 1.852 * 1000)
                # g = geod.Direct(lat1=self.sub_init_pos[id]['lat'], lon1=self.sub_init_pos[id]['lon'], azi1=course,
                #                 s12=d)
                # self.sub_target_pos[id]['lat'] = g['lat2']
                # self.sub_target_pos[id]['lon'] = g['lon2']

        for id in range(self.args.plane_nums):
            if id == 0:
                # 在大鲸级潜艇的附近，搜索钓鱼岛20-70海里
                i = 0
                course = np.random.uniform(-120, -160)
                d = np.random.uniform(3 * 1.852 * 1000,
                                      4 * 1.852 * 1000)  # 修改d = np.random.uniform(9 * 1.852 * 1000, 11 * 1.852 * 1000)
                g = geod.Direct(lat1=self.sub_init_pos[i]['lat'], lon1=self.sub_init_pos[i]['lon'], azi1=course,
                                s12=d)
                self.plane_init_pos[id]['lat'] = g['lat2']
                self.plane_init_pos[id]['lon'] = g['lon2']
                self.plane_init_pos[id]['height'] = self.args.plane_height

            if id == 1:
                # 搜索钓鱼岛西南海域60海里，海域半径100海里
                course = np.random.uniform(-150, -120)
                d = np.random.uniform(58 * 1.852 * 1000, 62 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.Diaoyu['lat'], lon1=self.args.Diaoyu['lon'], azi1=course, s12=d)
                self.plane_init_pos[id]['lat'] = g['lat2']
                self.plane_init_pos[id]['lon'] = g['lon2']
                self.plane_init_pos[id]['height'] = self.args.plane_height

        for id in range(self.args.red_sub_nums):
            if id == 1:
                course = np.random.uniform(30, 60)
                d = np.random.uniform(58 * 1.852 * 1000, 62 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.Huangwei['lat'], lon1=self.args.Huangwei['lon'], azi1=course, s12=d)
                self.red_sub_init_pos[id]['lat'] = g['lat2']
                self.red_sub_init_pos[id]['lon'] = g['lon2']
                self.red_sub_init_pos[id]['height'] = self.args.plane_height

        for id in range(self.args.red_frigate_nums):  # 护卫舰
            if id < 2:
                course = np.random.uniform(-10, 10)
                d = np.random.uniform(48 * 1.852 * 1000, 52 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.Huangwei['lat'], lon1=self.args.Huangwei['lon'], azi1=course, s12=d)
                self.red_frigate_init_pos[id]['lat'] = g['lat2']
                self.red_frigate_init_pos[id]['lon'] = g['lon2']
            else:
                course = np.random.uniform(-30, -60)
                d = np.random.uniform(48 * 1.852 * 1000, 52 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.Diaoyu['lat'], lon1=self.args.Diaoyu['lon'], azi1=course, s12=d)
                self.red_frigate_init_pos[id]['lat'] = g['lat2']
                self.red_frigate_init_pos[id]['lon'] = g['lon2']

        for id in range(self.args.red_maritime_ship_nums):  # 海警船
            if id == 0:
                course = np.random.uniform(-10, 10)
                d = np.random.uniform(48 * 1.852 * 1000, 52 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.Huangwei['lat'], lon1=self.args.Huangwei['lon'], azi1=course, s12=d)
                self.red_maritime_ship_init_pos[id]['lat'] = g['lat2']
                self.red_maritime_ship_init_pos[id]['lon'] = g['lon2']
            if id == 1:
                course = np.random.uniform(-30, -60)
                d = np.random.uniform(48 * 1.852 * 1000, 52 * 1.852 * 1000)
                g = geod.Direct(lat1=self.args.Diaoyu['lat'], lon1=self.args.Diaoyu['lon'], azi1=course, s12=d)
                self.red_maritime_ship_init_pos[id]['lat'] = g['lat2']
                self.red_maritime_ship_init_pos[id]['lon'] = g['lon2']

        # 蓝方巡逻舰位于南小岛东南方向，海域中心距离该岛40海里，海域半径20海里
        course = np.random.uniform(110, 160)
        d = np.random.uniform(38 * 1.852 * 1000, 42 * 1.852 * 1000)
        g = geod.Direct(lat1=self.args.Nanxiao['lat'], lon1=self.args.Nanxiao['lon'], azi1=course, s12=d)
        centor_pos = {"lat": g['lat2'], "lon": g['lon2']}
        for id in range(self.args.blue_patrol_ship_nums):
            course = np.random.uniform(0, 360)
            d = np.random.uniform(8 * 1.852 * 1000, 10 * 1.852 * 1000)
            g = geod.Direct(lat1=centor_pos['lat'], lon1=centor_pos['lon'], azi1=course, s12=d)
            self.blue_patrol_ship_init_pos[id]['lat'], self.blue_patrol_ship_init_pos[id]['lon'] = g['lat2'], g['lon2']

        # 蓝方驱逐舰位于赤尾岛以南方向，海域中心距离该岛20海里，海域半径6海里
        course = np.random.uniform(165, 195)
        d = np.random.uniform(18 * 1.852 * 1000, 22 * 1.852 * 1000)
        g = geod.Direct(lat1=self.args.Chiwei['lat'], lon1=self.args.Chiwei['lon'], azi1=course, s12=d)
        centor_pos = {"lat": g['lat2'], "lon": g['lon2']}
        for id in range(self.args.blue_destroyer_nums):
            course = np.random.uniform(0, 360)
            d = np.random.uniform(4 * 1.852 * 1000, 8 * 1.852 * 1000)
            g = geod.Direct(lat1=centor_pos['lat'], lon1=centor_pos['lon'], azi1=course, s12=d)
            self.blue_destroyer_init_pos[id]['lat'], self.blue_destroyer_init_pos[id]['lon'] = g['lat2'], g['lon2']

        # 蓝方P1飞机位于钓鱼岛60海里的上空、第一岛屿上空巡逻警戒
        course = np.random.uniform(-30, -60)
        d = np.random.uniform(58 * 1.852 * 1000, 62 * 1.852 * 1000)
        g = geod.Direct(lat1=self.args.Diaoyu['lat'], lon1=self.args.Diaoyu['lon'], azi1=course, s12=d)
        for id in range(self.args.blue_P1_plane_nums):
            self.blue_P1_plane_init_pos[id]['lat'], self.blue_P1_plane_init_pos[id]['lon'] = g['lat2'], g['lon2']

    def init_obs(self):
        obs_message = {"red_message": {"uav_message": [[] for _ in range(self.args.uav_nums)],
                                       "usv_message": [[] for _ in range(self.args.usv_nums)]},
                       "blue_message": {"sub_message": [[] for _ in range(self.args.submarine_nums)]},
                       "env_message": []}
        blue_obs = BlueGlobalObservation(self.args)
        red_obs = RedGlobalObservation(self.args)

        self.uav_state_change = [[] for _ in range(self.args.uav_nums)]  # 记录无人机速度变化及相关描述
        self.uav_state_class = [0 for _ in range(self.args.uav_nums)]

        # 潜艇初始化
        self.get_init_pos()  # 智能体位置初始化
        env_obs = EnvGlobalObservation(self.args)
        for i in range(blue_obs.submarine_nums):
            blue_obs.submarines[i].lat = self.sub_init_pos[i]["lat"]
            blue_obs.submarines[i].lon = self.sub_init_pos[i]["lon"]
            blue_obs.submarines[i].height = self.sub_init_pos[i]["height"]
            self.sub_pre_pos.append([blue_obs.submarines[i].lat, blue_obs.submarines[i].lon])
            blue_obs.submarines[i].vel = self.args.sub_vel
            blue_obs.submarines[i].height = self.sub_init_pos[i]["height"]
            blue_obs.submarines[i].course = 90
            blue_obs.submarines[i].type = self.args.sub_name[i]
            blue_obs.submarines[i].update_params()
            if i == 0:
                blue_obs.submarines[i].sub_field = self.args.blue_sub_field

        for id in range(self.args.blue_patrol_ship_nums):
            blue_obs.patrol_ships[id].lat = self.blue_patrol_ship_init_pos[id]["lat"]
            blue_obs.patrol_ships[id].lon = self.blue_patrol_ship_init_pos[id]["lon"]
            blue_obs.patrol_ships[id].course = 90
            blue_obs.patrol_ships[id].vel = self.args.blue_patrol_ship_vel
            blue_obs.patrol_ships[id].type = self.args.blue_patrol_ship_name
            blue_obs.patrol_ships[id].update_params()

        for id in range(self.args.blue_destroyer_nums):
            blue_obs.destroyers[id].lat = self.blue_destroyer_init_pos[id]["lat"]
            blue_obs.destroyers[id].lon = self.blue_destroyer_init_pos[id]["lon"]
            blue_obs.destroyers[id].course = 90
            blue_obs.destroyers[id].vel = self.args.blue_destroyer_vel
            blue_obs.destroyers[id].type = self.args.blue_destroyer_name[id]
            blue_obs.destroyers[id].update_params()

        for id in range(self.args.blue_P1_plane_nums):
            blue_obs.P1_plane[id].lat = self.blue_P1_plane_init_pos[id]["lat"]
            blue_obs.P1_plane[id].lon = self.blue_P1_plane_init_pos[id]["lon"]
            blue_obs.P1_plane[id].height = self.args.blue_P1_plane_height
            blue_obs.P1_plane[id].course = 90
            blue_obs.P1_plane[id].vel = self.args.blue_P1_plane_vel
            blue_obs.P1_plane[id].type = self.args.blue_P1_plane_name
            blue_obs.P1_plane[id].update_params()

        for id in range(self.args.blue_F15_plane_nums):
            blue_obs.F15_plane[id].lat = self.blue_F15_plane_init_pos[id]["lat"]
            blue_obs.F15_plane[id].lon = self.blue_F15_plane_init_pos[id]["lon"]
            blue_obs.F15_plane[id].type = self.args.blue_F15_plane_name
            blue_obs.F15_plane[id].course = 90
            blue_obs.F15_plane[id].update_params()

        for id in range(self.args.blue_Elec_plane_nums):
            blue_obs.Elec_plane[id].lat = self.blue_Elec_plane_init_pos[id]["lat"]
            blue_obs.Elec_plane[id].lon = self.blue_Elec_plane_init_pos[id]["lon"]
            blue_obs.Elec_plane[id].type = self.args.blue_Elec_plane_name
            blue_obs.Elec_plane[id].course = 90
            blue_obs.Elec_plane[id].update_params()

        for id in range(red_obs.uav_field.uav_nums):
            red_obs.uav_field.uavs[id].lat = self.uav_init_pos[id]['lat']
            red_obs.uav_field.uavs[id].lon = self.uav_init_pos[id]['lon']
            red_obs.uav_field.uavs[id].height = self.uav_init_pos[id]['height']
            red_obs.uav_field.uavs[id].vel = self.args.uav_vel
            red_obs.uav_field.uavs[id].course = 90
            red_obs.uav_field.uavs[id].type = self.args.uav_name
            red_obs.uav_field.uavs[id].update_params()

        for id in range(red_obs.usv_field.usv_nums):
            red_obs.usv_field.usvs[id].lat = self.usv_init_pos[id]["lat"]
            red_obs.usv_field.usvs[id].lon = self.usv_init_pos[id]["lon"]
            red_obs.usv_field.usvs[id].height = self.usv_init_pos[id]['height']
            red_obs.usv_field.usvs[id].vel = self.args.usv_vel
            red_obs.usv_field.usvs[id].type = self.args.usv_name
            red_obs.usv_field.usvs[id].course = 90
            red_obs.usv_field.usvs[id].update_params()

        for id in range(red_obs.plane_field.plane_nums):
            red_obs.plane_field.planes[id].lat = self.plane_init_pos[id]['lat']
            red_obs.plane_field.planes[id].lon = self.plane_init_pos[id]['lon']
            red_obs.plane_field.planes[id].height = self.plane_init_pos[id]['height']
            red_obs.plane_field.planes[id].vel = self.args.plane_vel
            red_obs.plane_field.planes[id].type = self.args.plane_name
            red_obs.plane_field.planes[id].course = 90
            red_obs.plane_field.planes[id].update_params()

        # 潜艇任务目标点
        # for id in range(self.args.submarine_nums):
        #     blue_obs.task_point[id].lat = self.sub_target_pos[id]['lat']
        #     blue_obs.task_point[id].lon = self.sub_target_pos[id]['lon']
        #     g = geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, blue_obs.task_point[id].lat,
        #                      blue_obs.task_point[id].lon)
        #     blue_obs.submarines[id].max_dis = g['s12']
        #     obs_message["blue_message"]["sub_message"][id].append({"title": "潜艇向目标点移动", "detail": "潜艇距离目标点距离为{}米".format(
        #                                                               round(blue_obs.submarines[id].max_dis, 2)),
        #                                                           'time': red_obs.simtime})

        for id in range(self.args.red_frigate_nums):
            red_obs.frigate_field[id].lat = self.red_frigate_init_pos[id]['lat']
            red_obs.frigate_field[id].lon = self.red_frigate_init_pos[id]['lon']
            red_obs.frigate_field[id].height = 0
            red_obs.frigate_field[id].vel = self.args.red_frigate_vel
            red_obs.frigate_field[id].type = self.args.red_frigate_name[id]
            red_obs.frigate_field[id].course = 90
            red_obs.frigate_field[id].update_params()

        for id in range(self.args.red_maritime_ship_nums):
            red_obs.maritime_ship_field[id].lat = self.red_maritime_ship_init_pos[id]['lat']
            red_obs.maritime_ship_field[id].lon = self.red_maritime_ship_init_pos[id]['lon']
            red_obs.maritime_ship_field[id].height = 0
            red_obs.maritime_ship_field[id].vel = self.args.red_maritime_ship_vel
            red_obs.maritime_ship_field[id].type = self.args.red_maritime_ship_name
            red_obs.maritime_ship_field[id].course = 90
            red_obs.maritime_ship_field[id].update_params()

        for id in range(self.args.red_J20_nums):
            red_obs.J20_field[id].lat = self.args.red_airport['lat']
            red_obs.J20_field[id].lon = self.args.red_airport['lon']
            red_obs.J20_field[id].type = self.args.red_J20_name
            red_obs.J20_field[id].course = 90
            red_obs.J20_field[id].update_params()

        for id in range(self.args.red_H6_nums):
            red_obs.H6_field[id].lat = self.args.red_airport['lat']
            red_obs.H6_field[id].lon = self.args.red_airport['lon']
            red_obs.H6_field[id].type = self.args.red_H6_name
            red_obs.H6_field[id].course = 90
            red_obs.H6_field[id].update_params()

        for id in range(self.args.red_Elec_plane_nums):
            red_obs.Elec_plane_field[id].lat = self.args.red_airport['lat']
            red_obs.Elec_plane_field[id].lon = self.args.red_airport['lon']
            red_obs.Elec_plane_field[id].type = self.args.red_Elec_plane_name
            red_obs.Elec_plane_field[id].course = 90
            red_obs.Elec_plane_field[id].update_params()

        for id in range(self.args.red_sub_nums):  # 红方潜艇位置初始化
            red_obs.red_sub_field.red_subs[id].type = self.args.red_sub_name
            if id == 0:
                lon = np.random.uniform(
                    self.args.blue_sub_field['min_lon'] * 2 / 3 + self.args.blue_sub_field['max_lon'] / 3,
                    self.args.blue_sub_field['min_lon'] / 3 + self.args.blue_sub_field['max_lon'] * 2 / 3)
                lat = self.args.blue_sub_field['max_lat'] + np.random.uniform(0.02, 0.03)
                red_obs.red_sub_field.red_subs[id].lat = lat
                red_obs.red_sub_field.red_subs[id].lon = lon
                red_obs.red_sub_field.red_subs[id].vel = self.args.red_sub_vel
                red_obs.red_sub_field.red_subs[id].height = self.args.red_sub_height
                lon = np.random.uniform(self.args.blue_sub_field['min_lon'], self.args.blue_sub_field['max_lon'])
                lat = self.args.blue_sub_field['min_lat'] - np.random.uniform(0.02, 0.03)
                red_obs.red_sub_field.red_subs[id].task_area.lat = lat
                red_obs.red_sub_field.red_subs[id].task_area.lon = lon
                g = geod.Inverse(red_obs.red_sub_field.red_subs[id].lat, red_obs.red_sub_field.red_subs[id].lon,
                                 red_obs.red_sub_field.red_subs[id].task_area.lat,
                                 red_obs.red_sub_field.red_subs[id].task_area.lon)
                red_obs.red_sub_field.red_subs[id].course = g['azi1']
                red_obs.red_sub_field.red_subs[id].update_params()
            if id == 1:
                red_obs.red_sub_field.red_subs[id].lat = self.red_sub_init_pos[id]['lat']
                red_obs.red_sub_field.red_subs[id].lon = self.red_sub_init_pos[id]['lon']
                red_obs.red_sub_field.red_subs[id].height = self.red_sub_init_pos[id]['height']
                red_obs.red_sub_field.red_subs[id].vel = self.args.red_sub_vel
                lon = np.random.uniform(self.args.blue_sub_field['min_lon'], self.args.blue_sub_field['max_lon'])
                lat = np.random.uniform(self.args.blue_sub_field['min_lat'], self.args.blue_sub_field['max_lat'])
                red_obs.red_sub_field.red_subs[id].task_area.lat = lat
                red_obs.red_sub_field.red_subs[id].task_area.lon = lon
                g = geod.Inverse(red_obs.red_sub_field.red_subs[id].lat, red_obs.red_sub_field.red_subs[id].lon,
                                 red_obs.red_sub_field.red_subs[id].task_area.lat,
                                 red_obs.red_sub_field.red_subs[id].task_area.lon)
                red_obs.red_sub_field.red_subs[id].course = g['azi1']
                red_obs.red_sub_field.red_subs[id].update_params()

            ############# 初始化飞机控制器 ##############
        self.plane_controls = []
        for uav in red_obs.uav_field.uavs:
            self.plane_controls.append(
                PlaneControl(lat=uav.lat, lon=uav.lon, course=uav.course, vel=uav.vel, alt=uav.height))

        self.sub_controls = []
        for submarine in blue_obs.submarines:
            self.sub_controls.append(
                SubControl(lat=submarine.lat, lon=submarine.lon, course=submarine.course, vel=submarine.vel,
                           height=submarine.height))

        self.blue_patrol_ship_controls = []
        for patrol_ship in blue_obs.patrol_ships:
            self.blue_patrol_ship_controls.append(
                UsvControl(lat=patrol_ship.lat, lon=patrol_ship.lon, course=patrol_ship.course, vel=patrol_ship.vel))

        self.blue_destroyer_controls = []
        for destroyer in blue_obs.destroyers:
            self.blue_destroyer_controls.append(
                UsvControl(lat=destroyer.lat, lon=destroyer.lon, course=destroyer.course, vel=destroyer.vel))

        self.blue_P1_plane_controls = []
        for P1 in blue_obs.P1_plane:
            self.blue_P1_plane_controls.append(
                PlaneControl(lat=P1.lat, lon=P1.lon, course=P1.course, vel=P1.vel, alt=P1.height))

        self.usv_controls = []
        for usv in red_obs.usv_field.usvs:
            self.usv_controls.append(UsvControl(lat=usv.lat, lon=usv.lon, course=usv.course, vel=usv.vel))

        self.Y9plane_controls = []
        for plane in red_obs.plane_field.planes:
            self.Y9plane_controls.append(
                PlaneControl(lat=plane.lat, lon=plane.lon, course=plane.course, vel=plane.vel, alt=plane.height,
                             max_alt=11000, max_vel=650))

        self.red_sub_controls = []
        for submarine in red_obs.red_sub_field.red_subs:
            self.red_sub_controls.append(
                SubControl(lat=submarine.lat, lon=submarine.lon, course=submarine.course, vel=submarine.vel,
                           height=submarine.height))

        self.frigate_controls = []
        for frigate in red_obs.frigate_field:
            self.frigate_controls.append(
                UsvControl(lat=frigate.lat, lon=frigate.lon, course=frigate.course, vel=frigate.vel))

        self.maritime_controls = []
        for maritime in red_obs.maritime_ship_field:
            self.maritime_controls.append(
                UsvControl(lat=maritime.lat, lon=maritime.lon, course=maritime.course, vel=maritime.vel))

        message = {"title": "任务海空域态势环境初始化", "detail": '完成双方任务海空域兵力兵器部署及环境数据初始化',
                   'time': blue_obs.simtime}
        obs_message["env_message"].append(message) if message not in obs_message["env_message"] else None

        # print("飞机与潜艇的距离", geod.Inverse(red_obs.uav_field.uavs[0].lat,red_obs.uav_field.uavs[0].lon, blue_obs.submarines[0].lat, blue_obs.submarines[0].lon)['s12'])
        # print("y9飞机与潜艇的距离", geod.Inverse(red_obs.plane_field.planes[0].lat, red_obs.plane_field.planes[0].lon, blue_obs.submarines[0].lat,
        #                    blue_obs.submarines[0].lon)['s12'])

        return red_obs, blue_obs, env_obs, obs_message

    def reset(self):
        self.red_obs, self.blue_obs, self.env_obs, obs_message = self.init_obs()
        # 信息初始化
        self.mode = [0 for _ in range(self.args.submarine_nums)]  # 0 -- 检查SQ， 1--应召SQ

        # 仿真结果统计
        self.result_statistics = {}
        # 蓝方仿真结果
        self.sub_pre_pos = [[0, 0] for _ in range(self.args.submarine_nums)]  # 潜艇上一时刻位置
        self.sub_low_speed_sailing_duration = [0 for _ in range(self.args.submarine_nums)]  # 低速航行时长(速度小于5节)，小时
        self.sub_high_speed_sailing_duration = [0 for _ in range(self.args.submarine_nums)]  # 高速航行时长(速度大于6节)，小时
        self.sub_goal_completion_rate = [0 for _ in range(self.args.submarine_nums)]  # 目标完成度
        self.sub_total_navigation_mileage = [0 for _ in range(self.args.submarine_nums)]  # 航行总里程，米
        self.sub_initial_exposure_time = 0  # 初始暴露时间
        self.sub_bait_deployed_num = 0  # 声诱饵投放数量
        self.sub_jammer_deployed_num = 0  # 干扰器投放数量
        self.sub_velocity_list = [[] for _ in range(self.args.submarine_nums)]  # 潜艇速度记录
        self.sub_action_list = [[] for _ in range(self.args.submarine_nums)]  # 潜艇动作记录
        self.sub_state_list = [[] for _ in range(self.args.submarine_nums)]  # 潜艇状态记录
        # 红方仿真结果
        # 无人机
        self.uav_pre_pos = [[0, 0] for _ in range(self.args.uav_nums)]  # 无人艇上一时刻位置
        self.uav_total_duration_call_point = [0 for _ in range(self.args.uav_nums)]  # 到达应召点总时长，小时
        self.uav_time_first_identified_sub = 0  # 第一次识别到潜艇花费时长，小时
        self.uav_total_navigation_mileage = [0 for _ in range(self.args.uav_nums)]  # 航行总里程，米
        self.uav_sonar_buoy_num = {"passive": 0, "activate": 0}  # 耗费声呐浮标数量
        self.uav_passive_sonar_survival_rate = 0  # 被动声呐存活率
        self.uav_active_sonar_survival_rate = 0  # 主动声呐存活率
        self.uav_target_recognition_accuracy = 0  # 目标识别准确率
        # 无人艇
        self.usv_pre_pos = [[0, 0] for _ in range(self.args.usv_nums)]  # 无人艇上一时刻位置
        self.usv_total_navigation_mileage = [0 for _ in range(self.args.usv_nums)]  # 航行总里程，米
        self.usv_total_duration_call_point = 0  # 到达应召点总时长，小时
        self.usv_time_first_identified_sub = 0  # 第一次识别到潜艇花费时长，小时
        self.blue_sub_info = [[] for _ in range(self.args.submarine_nums)]  # 蓝方潜艇角度信息

        self.red_report_infos = [[] for i in range(len(self.blue_obs.submarines))]

        return {"red_obs": self.red_obs,
                "blue_obs": self.blue_obs,
                "env_obs": self.env_obs,
                "obs_message": obs_message}

    def step(self, command_dict):
        sensor_data = False  # 是否获取传感器探测数据，False时只获得结果
        obs_message = {
            "red_message": {"uav_message": {i: [] for i in range(self.red_obs.uav_field.uav_nums)},
                            "usv_message": {i: [] for i in range(self.red_obs.usv_field.usv_nums)},
                            "plane_message": {i: [] for i in range(self.red_obs.plane_field.plane_nums)}},
            "blue_message": {"sub_message": {i: [] for i in range(self.blue_obs.submarine_nums)}},
            "env_message": []}
        key_message = []
        blue_cmds = command_dict["blue_cmds"]
        red_cmds = command_dict["red_cmds"]
        if len(command_dict["blue_message"]) > 0:
            obs_message["blue_message"]['sub_message'][0].extend(command_dict["blue_message"])
        if len(command_dict['blue_key_message']) > 0:
            key_message.extend(command_dict["blue_message"])

        # 获取应召点
        for target_id in range(self.blue_obs.submarine_nums):
            if self.mode[target_id] == 0:
                submarine = self.blue_obs.submarines[target_id]
                if not self.red_obs.call_point[target_id].lat:
                    for id in range(self.red_obs.plane_field.plane_nums):
                        plane = self.red_obs.plane_field.planes[id]
                        g = geod.Inverse(plane.lat, plane.lon, submarine.lat, submarine.lon)['s12']
                        if g < 5_000:
                            course = np.random.uniform(0, 360)
                            d = np.random.uniform(1000, 3000)
                            g = geod.Direct(lat1=submarine.lat, lon1=submarine.lon, azi1=course, s12=d)
                            self.red_obs.call_point[target_id].lat = g['lat2']
                            self.red_obs.call_point[target_id].lon = g['lon2']
                            self.mode[target_id] = 1
                            message = {"title": "红方获得蓝方潜艇应召点信息",
                                       "detail": f'红方{id + 1}号运九无人机获得蓝方{target_id + 1}号潜艇的应召点信息，应召点经度为{self.red_obs.call_point[target_id].lon}、纬度为{self.red_obs.call_point[target_id].lat}',
                                       'time': self.blue_obs.simtime}
                            obs_message["red_message"]["plane_message"][id].append(message) if message not in \
                                                                                               obs_message[
                                                                                                   "red_message"][
                                                                                                   "plane_message"][
                                                                                                   id] else None
                            message = {"info": "红方获得蓝方潜艇应召点信息", "class": 'red', "type": 'Y9_plane',
                                       'id': id}
                            key_message.append(message) if message not in key_message else None
                            break

        result = Result.GO_ON
        self.red_obs.report.clear_info()  # 上报信息清空

        for blue_cmd in blue_cmds:
            type = blue_cmd["type"]
            id = blue_cmd["id"]
            obj_type = blue_cmd["obj_type"]

            # 移动
            if type == CmdType.MOVE:
                obj = None
                if obj_type == BlueObjType.SUBMARINE:
                    obj = self.blue_obs.submarines[id]

                    obj.course, obj.vel, obj.height = self.sub_controls[id].control(
                        course=obj.course if blue_cmd["course"] is None else blue_cmd["course"],
                        vel=obj.vel if blue_cmd["vel"] is None else blue_cmd["vel"],
                        height=obj.height if blue_cmd["height"] is None else blue_cmd["height"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel * 1852 / 3600)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新潜艇行驶里程
                    obj.update_params()
                    # print('sub vel', obj.vel)

                if obj_type == BlueObjType.PATROL:
                    obj = self.blue_obs.patrol_ships[id]
                    obj.course, obj.vel = self.blue_patrol_ship_controls[id].control(
                        course=obj.course if blue_cmd["course"] is None else blue_cmd["course"],
                        vel=obj.vel if blue_cmd["vel"] is None else blue_cmd["vel"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel * 1852 / 3600)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.height = np.random.uniform(-0.5, -0.2)
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新行驶里程
                    obj.update_params()

                if obj_type == BlueObjType.DESTROYER:
                    obj = self.blue_obs.destroyers[id]
                    obj.course, obj.vel = self.blue_destroyer_controls[id].control(
                        course=obj.course if blue_cmd["course"] is None else blue_cmd["course"],
                        vel=obj.vel if blue_cmd["vel"] is None else blue_cmd["vel"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel * 1852 / 3600)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.height = np.random.uniform(-0.5, -0.2)
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新行驶里程
                    obj.update_params()

                if obj_type == BlueObjType.P1:
                    obj = self.blue_obs.P1_plane[id]
                    obj.course, obj.vel, obj.height = self.blue_P1_plane_controls[id].control(
                        course=obj.course if blue_cmd["course"] is None else blue_cmd["course"],
                        vel=obj.vel if blue_cmd["vel"] is None else blue_cmd["vel"],
                        height=obj.height if blue_cmd["height"] is None else blue_cmd["height"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel / 3.6)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新行驶里程
                    obj.update_params()

            if type == CmdType.Jammer:
                if self.args.Sub_params_Init_jammer:  # 是否携带了干扰器
                    height = blue_cmd["height"]
                    lat = blue_cmd["lat"]
                    lon = blue_cmd["lon"]
                    id = blue_cmd["id"]
                    obj = self.blue_obs.submarines[id]
                    if obj.jammer_nums > 0:
                        obj.jammers.append(Jammer(lat=lat, lon=lon, height=height, start_time=self.blue_obs.simtime))
                        obj.jammer_nums -= 1
                        self.sub_jammer_deployed_num += 1
                        self.sub_action_list[id].append("潜艇投放干扰器")
                        self.sub_state_list[id].append("潜艇折线躲避")
                        message = {"title": "潜艇投放干扰器",
                                   "detail": f'{id + 1}号潜艇投放干扰器，投放位置的经度为{lon}、纬度为{lat}，深度为{height}',
                                   'time': self.blue_obs.simtime}
                        obs_message["blue_message"]["sub_message"][id].append(message) if message not in \
                                                                                          obs_message["blue_message"][
                                                                                              "sub_message"][
                                                                                              id] else None
                        message = {"info": "蓝方潜艇投放干扰器", "class": 'blue', "type": 'submarine', 'id': id}
                        key_message.append(message) if message not in key_message else None
                else:
                    raise ValueError(f"潜艇{id}没有携带干扰器")

            # 声诱饵
            if type == CmdType.BAIT:
                if self.args.Sub_params_Init_Bait:
                    height = blue_cmd["height"]
                    lat = blue_cmd["lat"]
                    lon = blue_cmd["lon"]
                    id = blue_cmd["id"]
                    velocity = blue_cmd["velocity"]
                    course = blue_cmd["course"]
                    obj = self.blue_obs.submarines[id]
                    if obj.bait_nums > 0:
                        obj.bait.append(acoustic_bait(lat=lat, lon=lon, height=height, velocity=velocity, course=course,
                                                      start_time=self.blue_obs.simtime))
                        obj.bait_nums -= 1
                        self.sub_bait_deployed_num += 1
                        self.sub_action_list[id].append("潜艇投放声诱饵")
                        message = {"title": "潜艇投放声诱饵",
                                   "detail": f'{id + 1}号潜艇投放干扰器，投放位置的经度为{lon}、纬度为{lat}，深度为{height}',
                                   'time': self.blue_obs.simtime}
                        obs_message["blue_message"]["sub_message"][id].append(message) if message not in \
                                                                                          obs_message["blue_message"][
                                                                                              "sub_message"][
                                                                                              id] else None
                        message = {"info": "蓝方潜艇投放声诱饵", "class": 'blue', "type": 'submarine', 'id': id}
                        key_message.append(message) if message not in key_message else None
                else:
                    raise ValueError(f"潜艇{id}没有携带声诱饵")

            # 声呐
            if type == CmdType.sub_drag_sonar:
                # 潜艇打开拖曳声呐
                statu = blue_cmd["statu"]
                theta_rope = blue_cmd["theta_rope"]
                rope_len = blue_cmd["rope_len"]
                theta_hydrophone = blue_cmd["theta_hydrophone"]
                id = blue_cmd["id"]
                self.blue_obs.submarines[id].drag_sonar.sonar_control(statu=statu, theta_rope=theta_rope,
                                                                      rope_len=rope_len,
                                                                      theta_hydrophone=theta_hydrophone)

                if self.blue_obs.submarines[id].drag_sonar.open_time != self.blue_obs.simtime or \
                        self.blue_obs.submarines[id].drag_sonar.open_time == 0:
                    self.blue_obs.submarines[id].drag_sonar.open_time = self.blue_obs.simtime
                    message = {"title": "潜艇打开拖曳声呐",
                               "detail": f'{id + 1}号潜艇打开拖曳声呐，拖曳声呐绳缆长度为{rope_len}、绳缆与海平面夹角为{theta_rope}，拖曳阵与绳缆夹角为{theta_hydrophone}',
                               'time': self.blue_obs.simtime}
                    obs_message["blue_message"]["sub_message"][id].append(message) if message not in \
                                                                                      obs_message["blue_message"][
                                                                                          "sub_message"][id] else None
                self.blue_obs.submarines[id].drag_sonar.open_time += 1

            if type == CmdType.sub_sonar:
                self.blue_obs.submarines[id].sonar.statu = blue_cmd["statu"]
                if self.blue_obs.submarines[id].sonar.open_time != self.blue_obs.simtime or self.blue_obs.submarines[
                    id].sonar.open_time == 0:
                    self.blue_obs.submarines[id].sonar.open_time = self.blue_obs.simtime
                    message = {"title": "潜艇打开艇壳声呐", "detail": f'{id + 1}号潜艇打开艇壳声呐',
                               'time': self.blue_obs.simtime}
                    obs_message["blue_message"]["sub_message"][id].append(message) if message not in \
                                                                                      obs_message["blue_message"][
                                                                                          "sub_message"][id] else None
                self.blue_obs.submarines[id].sonar.open_time += 1

            if type == CmdType.Snorkel:
                if self.blue_obs.submarines[id].snorkel_open_time != self.blue_obs.simtime or self.blue_obs.submarines[
                    id].snorkel_open_time == 0:
                    self.blue_obs.submarines[id].snorkel_open_time = self.blue_obs.simtime
                    message = {"title": "潜艇打开通气管", "detail": f'{id + 1}号潜艇打开通气管',
                               'time': self.blue_obs.simtime}
                    obs_message["blue_message"]["sub_message"][id].append(message) if message not in \
                                                                                      obs_message["blue_message"][
                                                                                          "sub_message"][id] else None
                self.blue_obs.submarines[id].snorkel_open_time += 1

                if self.blue_obs.submarines[id].height > -15:
                    self.blue_obs.submarines[id].snorkel = blue_cmd["statu"]

            if type == CmdType.Periscope:
                self.blue_obs.submarines[id].periscope.statu = blue_cmd["statu"]
                if self.blue_obs.submarines[id].periscope.open_time != self.blue_obs.simtime or \
                        self.blue_obs.submarines[id].periscope.open_time == 0:
                    self.blue_obs.submarines[id].periscope.open_time = self.blue_obs.simtime
                    message = {"title": "潜艇打开潜望镜", "detail": f'{id + 1}号潜艇打开潜望镜',
                               'time': self.blue_obs.simtime}
                    obs_message["blue_message"]["sub_message"][id].append(message) if message not in \
                                                                                      obs_message["blue_message"][
                                                                                          "sub_message"][id] else None
                self.blue_obs.submarines[id].periscope.open_time += 1

        for i in range(self.blue_obs.submarine_nums):
            # 仿真结果统计
            self.sub_velocity_list[i].append(self.blue_obs.submarines[i].vel)
            if self.blue_obs.submarines[i].vel < 6:
                self.sub_low_speed_sailing_duration[i] += 1
            elif self.blue_obs.submarines[i].vel < 10:
                self.sub_routine_speed_sailing_duration[i] += 1
            else:
                self.sub_high_speed_sailing_duration[i] += 1
            self.sub_total_navigation_mileage[i] += \
            geod.Inverse(self.sub_pre_pos[i][0], self.sub_pre_pos[i][1], self.blue_obs.submarines[i].lat,
                         self.blue_obs.submarines[i].lon)['s12']
            self.sub_pre_pos[i] = [self.blue_obs.submarines[i].lat, self.blue_obs.submarines[i].lon]
            if len(self.sub_action_list[i]) == 0 or self.sub_action_list[i][-1] == "":
                self.sub_action_list[i].append("")
            if len(self.sub_state_list[i]) == 0 or self.sub_state_list[i][-1] == "":
                self.sub_state_list[i].append("")

            # 干扰器更新
            for jammer in self.blue_obs.submarines[i].jammers:
                jammer.update(self.blue_obs.simtime)  # 更新生存状态

            # 声诱饵更新
            for bait in self.blue_obs.submarines[i].bait:
                bait.update(self.blue_obs.simtime)  # 更新位置信息

            # 更新拖曳声呐信息
            if self.blue_obs.submarines[i].drag_sonar.statu and self.blue_obs.simtime % self.sonar_update_times == 0:
                self.blue_obs.submarines[i].drag_sonar.result_clear()
                self.blue_obs.submarines[i].drag_sonar.sensor_detect(self.red_obs, self.blue_obs, self.env_obs, id=i,
                                                                     thermocline_height=self.args.thermocline,
                                                                     sensor_data=sensor_data)
                if self.blue_obs.submarines[i].drag_sonar.touch:
                    if self.blue_obs.submarines[i].drag_sonar.touch_time != self.blue_obs.simtime or \
                            self.blue_obs.submarines[i].drag_sonar.touch_time == 0:
                        self.blue_obs.submarines[i].drag_sonar.touch_time = self.blue_obs.simtime
                        target_pos_info = ""
                        for j in range(len(self.blue_obs.submarines[i].drag_sonar.target_pos)):
                            target_pos_info += f'第{j + 1}个可疑目标的经度为{self.blue_obs.submarines[i].drag_sonar.target_pos[j]["lon"]}、纬度为{self.blue_obs.submarines[i].drag_sonar.target_pos[j]["lat"]}；'
                        target_pos_info = target_pos_info[:-1]
                        message = {"title": "潜艇拖曳声呐发现可疑目标",
                                   "detail": f"潜艇拖曳声呐发现{len(self.blue_obs.submarines[i].drag_sonar.target_pos)}个可疑目标, " + target_pos_info,
                                   'time': self.blue_obs.simtime}
                        obs_message["blue_message"]["sub_message"][i].append(message) if message not in \
                                                                                         obs_message["blue_message"][
                                                                                             "sub_message"][i] else None
                    self.blue_obs.submarines[i].drag_sonar.touch_time += self.sonar_update_times

            # 更新舰壳声呐信息
            if self.blue_obs.submarines[i].sonar.statu and self.blue_obs.simtime % self.sonar_update_times == 0:
                self.blue_obs.submarines[i].sonar.result_clear()
                self.blue_obs.submarines[i].sonar.sensor_detect(self.red_obs, self.blue_obs, self.env_obs,
                                                                thermocline_height=self.args.thermocline, id=i,
                                                                sensor_data=sensor_data)
                if self.blue_obs.submarines[i].sonar.touch:
                    if self.blue_obs.submarines[i].sonar.touch_time != self.blue_obs.simtime or \
                            self.blue_obs.submarines[i].sonar.touch_time == 0:
                        self.blue_obs.submarines[i].sonar.touch_time = self.blue_obs.simtime
                        target_pos_info = ""
                        for j in range(len(self.blue_obs.submarines[i].sonar.target_course)):
                            target_pos_info += f'第{j + 1}个可疑目标的方位角为{self.blue_obs.submarines[i].sonar.target_course[j]}、频率为{self.blue_obs.submarines[i].sonar.target_feature[j]["f"]}、信号声强级为{self.blue_obs.submarines[i].sonar.target_feature[j]["p_recevied"]}；'
                        target_pos_info = target_pos_info[:-1]
                        message = {"title": "潜艇舰壳声呐发现可疑目标",
                                   "detail": f"潜艇舰壳声呐发现{len(self.blue_obs.submarines[i].sonar.target_course)}个可疑目标, " + target_pos_info,
                                   'time': self.blue_obs.simtime}
                        obs_message["blue_message"]["sub_message"][i].append(message) if message not in \
                                                                                         obs_message["blue_message"][
                                                                                             "sub_message"][i] else None
                    self.blue_obs.submarines[i].sonar.touch_time += self.sonar_update_times

            self.blue_obs.submarines[i].battery.update_battery(self.blue_obs.submarines[i].vel * 1.852 / 3.6,
                                                               2 * 1.852 / 3.6,
                                                               snorkel=self.blue_obs.submarines[i].snorkel)  # 潜艇电量更新
            # 更新潜望镜信息
            if self.blue_obs.submarines[i].periscope.statu:
                self.blue_obs.submarines[i].periscope.result_clear()
                self.blue_obs.submarines[i].periscope.sensor_detect(self.red_obs, self.env_obs,
                                                                    self.blue_obs.submarines[i].lat,
                                                                    self.blue_obs.submarines[i].lon,
                                                                    self.blue_obs.submarines[i].height)
                if self.blue_obs.submarines[i].periscope.touch:
                    if self.blue_obs.submarines[i].periscope.touch_time != self.blue_obs.simtime or \
                            self.blue_obs.submarines[i].periscope.touch_time == 0:
                        self.blue_obs.submarines[i].periscope.touch_time = self.blue_obs.simtime
                        target_pos_info = ""
                        for j in range(len(self.blue_obs.submarines[i].periscope.result)):
                            target_type = list(self.blue_obs.submarines[i].periscope.result[j].keys())[0]
                            if target_type == 'uav':
                                target_type_ch = '无人机'
                            elif target_type == 'usv':
                                target_type_ch = '无人艇'
                            elif target_type == 'fishing_boat':
                                target_type_ch = '渔船'
                            else:
                                target_type_ch = '货轮'
                            target_pos_info += f'第{j + 1}个可疑目标为{target_type_ch}、经度为{self.blue_obs.submarines[i].periscope.result[j][target_type]["lon"]}、纬度为{self.blue_obs.submarines[i].periscope.result[j][target_type]["lat"]}、高度为{self.blue_obs.submarines[i].periscope.result[j][target_type]["height"]}；'
                        target_pos_info = target_pos_info[:-1]
                        message = {"title": "潜艇潜望镜发现可疑目标",
                                   "detail": f"潜艇舰壳声呐发现{len(self.blue_obs.submarines[i].periscope.result)}个可疑目标, " + target_pos_info,
                                   'time': self.blue_obs.simtime}
                        obs_message["blue_message"]["sub_message"][i].append(message) if message not in \
                                                                                         obs_message["blue_message"][
                                                                                             "sub_message"][i] else None
                    self.blue_obs.submarines[i].sonar.touch_time += 1

        for red_cmd in red_cmds:
            type = red_cmd["type"]
            id = red_cmd["id"]

            if type == CmdType.MOVE:
                obj_type = red_cmd["obj_type"]
                if obj_type == RedObjType.UAV:
                    obj = self.red_obs.uav_field.uavs[id]
                    obj.course, obj.vel, obj.height = self.plane_controls[id].control(
                        course=red_cmd.get('course', None),
                        height=red_cmd.get('height', None),
                        vel=red_cmd.get('vel', None)
                    )
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel / 3.6)

                    obj.lat = g['lat2']
                    obj.lon = g['lon2']

                elif obj_type == RedObjType.USV:
                    obj = self.red_obs.usv_field.usvs[id]
                    obj.course, obj.vel = self.usv_controls[id].control(
                        course=obj.course if red_cmd["course"] is None else red_cmd["course"],
                        vel=obj.vel if red_cmd["vel"] is None else red_cmd["vel"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel * 1852 / 3600)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.height = np.random.uniform(-0.5, -0.2)
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新usv行驶里程
                    obj.update_params()

                elif obj_type == RedObjType.PLANE:
                    obj = self.red_obs.plane_field.planes[id]
                    obj.course, obj.vel, obj.height = self.Y9plane_controls[id].control(
                        course=obj.course if red_cmd["course"] is None else red_cmd["course"],
                        height=obj.height if red_cmd["height"] is None else red_cmd["height"],
                        vel=obj.vel if red_cmd["vel"] is None else red_cmd["vel"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel / 3.6)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新PLANE行驶里程
                    obj.update_params()

                elif obj_type == RedObjType.SUB:
                    obj = self.red_obs.red_sub_field.red_subs[id]
                    obj.course, obj.vel, obj.height = self.red_sub_controls[id].control(
                        course=obj.course if red_cmd["course"] is None else red_cmd["course"],
                        vel=obj.vel if red_cmd["vel"] is None else red_cmd["vel"],
                        height=obj.height if red_cmd["height"] is None else red_cmd["height"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel * 1852 / 3600)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新红方潜艇行驶里程
                    obj.update_params()

                elif obj_type == RedObjType.FRIGATE:
                    obj = self.red_obs.frigate_field[id]
                    obj.course, obj.vel = self.frigate_controls[id].control(
                        course=obj.course if red_cmd["course"] is None else red_cmd["course"],
                        vel=obj.vel if red_cmd["vel"] is None else red_cmd["vel"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel * 1852 / 3600)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.height = np.random.uniform(-0.5, -0.2)
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新行驶里程
                    obj.update_params()
                elif obj_type == RedObjType.MARITIME:
                    obj = self.red_obs.maritime_ship_field[id]
                    obj.course, obj.vel = self.maritime_controls[id].control(
                        course=obj.course if red_cmd["course"] is None else red_cmd["course"],
                        vel=obj.vel if red_cmd["vel"] is None else red_cmd["vel"])
                    g = geod.Direct(lat1=obj.lat, lon1=obj.lon, azi1=obj.course, s12=obj.vel * 1852 / 3600)
                    obj.lat = g['lat2']
                    obj.lon = g['lon2']
                    obj.height = np.random.uniform(-0.5, -0.2)
                    obj.mileage += geod.Inverse(obj.last_lat, obj.last_lon, obj.lat, obj.lon)['s12']  # 更新行驶里程
                    obj.update_params()

                else:
                    raise ValueError('red obj error...')

            if type == CmdType.REPORT:
                target_id = red_cmd["target_id"]
                obj_type = red_cmd["obj_type"]
                if obj_type == RedObjType.UAV:
                    type = 'uav'
                    agent_info = '无人机'
                    message_class = 'uav_message'
                else:
                    type = 'usv'
                    agent_info = '无人艇'
                    message_class = 'usv_message'
                if self.red_obs.report_open_time % 50 == 0:
                    if all(element is None for element in self.red_obs.report.lat):
                        message_info = "红方开始上报蓝方潜艇信息"
                    else:
                        message_info = f"红方第{self.red_obs.report_open_time}次上报蓝方潜艇信息"
                    message = {"info": message_info, "class": 'red', "type": type, 'id': id}
                    key_message.append(message) if message not in key_message else None
                    self.red_obs.report_open_time += 1

                self.red_obs.report.lat[target_id], self.red_obs.report.lon[target_id], self.red_obs.report.height[
                    target_id], self.red_obs.report.course[target_id], self.red_obs.report.vel[target_id], \
                self.red_obs.report.report_time[target_id] = \
                    red_cmd["target_lat"], red_cmd["target_lon"], red_cmd["target_height"], red_cmd["target_course"], \
                    red_cmd["target_vel"], red_cmd["report_time"]
                self.red_obs.report.report_id[target_id][type].append(id)  # 记录是红方的谁上报的

                report_lat, report_lon, _, report_course, report_vel, report_simtime = red_cmd["target_lat"], red_cmd[
                    "target_lon"], red_cmd["target_height"], red_cmd["target_course"], red_cmd["target_vel"], red_cmd[
                    "report_time"]

                # if len(self.red_report_infos[target_id]) == 0:
                self.red_report_infos[target_id].append(ReportInfo(
                    simtime=report_simtime,
                    report_lat=report_lat,
                    report_lon=report_lon,
                    report_course=report_course,
                    report_vel=report_vel,
                    sub_lat=self.blue_obs.submarines[target_id].lat,
                    sub_lon=self.blue_obs.submarines[target_id].lon,
                    sub_course=self.blue_obs.submarines[target_id].course,
                    sub_vel=self.blue_obs.submarines[target_id].vel
                ))

                target = self.blue_obs.submarines[target_id]
                g = geod.Inverse(target.lat, target.lon, red_cmd["target_lat"], red_cmd["target_lon"])
                dis_error = g['s12']
                couse_error = red_cmd["target_course"] - target.course
                couse_error = couse_error + 360 if couse_error < 0 else couse_error
                couse_error = 360 - couse_error if couse_error > 180 else couse_error
                vel_error = abs(red_cmd["target_vel"] - target.vel)

                # with open("./report.txt", "a", encoding="utf-8") as f:
                #     info = str(f"simtime:{self.red_obs.simtime}" + ", " + f"dis_error:{dis_error}" + ", " + f"couse_error:{couse_error}" + ", " + f"vel_error:{vel_error}" + "\n")
                #     f.writelines(info)

                if sum(1 for x in self.red_obs.report.report_id[target_id][type] if x == id) % 50 == 0:
                    message = {"title": "红方上报蓝方潜艇信息",
                               "detail": f'{id + 1}号' + agent_info + f'第{sum(1 for x in self.red_obs.report.report_id[target_id][type] if x == id)}次上报蓝方潜艇信息，上报信息经度为{self.red_obs.report.lon[target_id]}、纬度为{self.red_obs.report.lat[target_id]}，角度为{self.red_obs.report.course[target_id]}度，速度为{self.red_obs.report.vel[target_id]}节',
                               'time': self.red_obs.simtime}
                    obs_message["red_message"][message_class][id].append(message) if message not in \
                                                                                     obs_message["red_message"][
                                                                                         message_class][id] else None

            if type == CmdType.PLACE_BUOY:
                if self.args.Uav_params_Buoys:
                    buoy_type = red_cmd["buoy_type"]
                    height = red_cmd["height"]
                    channel = red_cmd["channel"]
                    id = red_cmd["id"]
                    obj = self.red_obs.uav_field.uavs[id]

                    if buoy_type == 0:
                        if obj.buoy_passive_nums == 0:
                            pass
                        else:
                            obj.buoys.append(Buoy(btype=0, lat=obj.lat,
                                                  lon=obj.lon, channel=channel,
                                                  height=height, start_time=self.red_obs.simtime))
                            obj.buoy_passive_use += 1
                            obj.buoy_passive_nums -= 1
                            message = {"title": f"无人机投放声呐浮标",
                                       "detail": f'{id + 1}号无人机投放被动声呐浮标，当前该无人机已投放{obj.buoy_passive_use}个被动声呐浮标、{obj.buoy_activate_use}个主动声呐浮标，剩余被动声呐浮标{obj.buoy_passive_nums}个，剩余主动声呐浮标{obj.buoy_activate_nums}个',
                                       'time': self.red_obs.simtime}
                            obs_message["red_message"]["uav_message"][id].append(message) if message not in \
                                                                                             obs_message["red_message"][
                                                                                                 "uav_message"][
                                                                                                 id] else None

                    if buoy_type == 1:
                        if obj.buoy_activate_nums == 0:
                            pass
                        else:
                            obj.buoys.append(Buoy(btype=1, lat=obj.lat,
                                                  lon=obj.lon, channel=channel,
                                                  height=height, start_time=self.red_obs.simtime))
                            obj.buoy_activate_use += 1
                            obj.buoy_activate_nums -= 1
                            message = {"title": f"无人机投放声呐浮标",
                                       "detail": f'{id + 1}号无人机投放主动声呐浮标，当前该无人机已投放{obj.buoy_passive_use}个被动声呐浮标、{obj.buoy_activate_use}个主动声呐浮标，剩余被动声呐浮标{obj.buoy_passive_nums}个，剩余主动声呐浮标{obj.buoy_activate_nums}个',
                                       'time': self.red_obs.simtime}
                            obs_message["red_message"]["uav_message"][id].append(message) if message not in \
                                                                                             obs_message["red_message"][
                                                                                                 "uav_message"][
                                                                                                 id] else None
                else:
                    raise ValueError(f"无人机{id}没有搭载声纳吊舱")

            if type == CmdType.INFRARED:
                if self.args.Uav_params_Infrared:  # 控制红外传感器
                    statu = red_cmd["statu"]
                    self.red_obs.uav_field.uavs[id].infrared.statu = statu
                    if self.red_obs.uav_field.uavs[id].infrared.open_time != self.red_obs.simtime or \
                            self.red_obs.uav_field.uavs[id].infrared.open_time == 0:
                        self.red_obs.uav_field.uavs[id].infrared.open_time = self.red_obs.simtime
                        message = {"title": f"无人机打开红外传感器", "detail": f'{id + 1}号无人机打开红外传感器',
                                   'time': self.red_obs.simtime}
                        obs_message["red_message"]["uav_message"][id].append(message) if message not in \
                                                                                         obs_message["red_message"][
                                                                                             "uav_message"][
                                                                                             id] else None
                    self.red_obs.uav_field.uavs[id].infrared.open_time += 1
                else:
                    raise ValueError(f"无人机{id}没有搭载红外成像仪")

            if type == CmdType.MAG:
                if self.args.Uav_params_Magnetic:  # 控制磁探传感器
                    statu = red_cmd["statu"]
                    self.red_obs.uav_field.uavs[id].magnetic.statu = statu
                    if self.red_obs.uav_field.uavs[id].magnetic.open_time != self.red_obs.simtime or \
                            self.red_obs.uav_field.uavs[id].magnetic.open_time == 0:
                        self.red_obs.uav_field.uavs[id].magnetic.open_time = self.red_obs.simtime
                        message = {"title": f"无人机打开磁探传感器", "detail": f'{id + 1}号无人机打开磁探传感器',
                                   'time': self.red_obs.simtime}
                        obs_message["red_message"]["uav_message"][id].append(message) if message not in \
                                                                                         obs_message["red_message"][
                                                                                             "uav_message"][
                                                                                             id] else None
                    self.red_obs.uav_field.uavs[id].magnetic.open_time += 1
                else:
                    raise ValueError(f"无人机{id}没有搭载磁探仪")

            if type == CmdType.USV_sonar:
                if self.args.Usv_params_Sonar:  # 控制拖拽传感器
                    if self.red_obs.usv_field.usvs[id].sonar.open_time != self.red_obs.simtime or \
                            self.red_obs.usv_field.usvs[id].sonar.open_time == 0:
                        self.red_obs.usv_field.usvs[id].sonar.open_time = self.red_obs.simtime
                        message = {"title": f"无人艇打开拖拽声呐传感器",
                                   "detail": f"{id + 1}号无人艇打开拖拽声呐传感器", 'time': self.blue_obs.simtime}
                        obs_message["red_message"]["usv_message"][id].append(message) if message not in \
                                                                                         obs_message["red_message"][
                                                                                             "usv_message"][
                                                                                             id] else None
                    self.red_obs.usv_field.usvs[id].sonar.open_time += 1

                    g = geod.Inverse(self.red_obs.usv_field.usvs[id].lat, self.red_obs.usv_field.usvs[id].lon,
                                     self.blue_obs.submarines[0].lat, self.blue_obs.submarines[0].lon)  # 为了加快速度，
                    if self.red_obs.usv_field.usvs[id].vel <= 20 and g['s12'] < 8000:
                        statu = red_cmd["statu"]
                        theta_rope = red_cmd["theta_rope"]
                        rope_len = red_cmd["rope_len"]
                        theta_hydrophone = red_cmd["theta_hydrophone"]
                        self.red_obs.usv_field.usvs[id].sonar.sonar_control(statu=statu, theta_rope=theta_rope,
                                                                            rope_len=rope_len,
                                                                            theta_hydrophone=theta_hydrophone)
                else:
                    raise ValueError(f"无人艇{id}没有搭载拖拽传感器")

        for id in range(self.red_obs.usv_field.usv_nums):  # 记录无人艇信息
            self.usv_total_navigation_mileage[id] += \
            geod.Inverse(self.usv_pre_pos[id][0], self.usv_pre_pos[id][1], self.red_obs.usv_field.usvs[id].lat,
                         self.red_obs.usv_field.usvs[id].lon)['s12']
            self.usv_pre_pos[id] = [self.red_obs.usv_field.usvs[id].lat, self.red_obs.usv_field.usvs[id].lon]

        for id in range(self.red_obs.uav_field.uav_nums):
            obj = self.red_obs.uav_field.uavs[id]
            self.red_obs.uav_field.uavs[id].mileage += \
            geod.Inverse(self.red_obs.uav_field.uavs[id].last_lat, self.red_obs.uav_field.uavs[id].last_lon, obj.lat,
                         obj.lon)['s12']
            self.red_obs.uav_field.uavs[id].update_params()
            # 记录每架飞机飞行总里程
            self.uav_total_navigation_mileage[id] = self.red_obs.uav_field.uavs[id].mileage

            # 记录无人机速度
            if command_dict['result']['uav_speed_record'][id]['start_time']:
                # 飞机开始运动
                if len(self.uav_state_change[id]) == self.uav_state_class[id]:
                    self.uav_state_change[id].append(
                        {f'describe{self.uav_state_class[id]}': {'vel': [], 'height': [], 'time': []}})

                self.uav_state_change[id][self.uav_state_class[id]][f'describe{self.uav_state_class[id]}'][
                    'vel'].append(obj.vel)
                self.uav_state_change[id][self.uav_state_class[id]][f'describe{self.uav_state_class[id]}'][
                    'height'].append(obj.height)
                self.uav_state_change[id][self.uav_state_class[id]][f'describe{self.uav_state_class[id]}'][
                    'time'].append(self.red_obs.simtime)
                if command_dict['result']['uav_speed_record'][id]['end_time']:
                    if self.red_obs.simtime == command_dict['result']['uav_speed_record'][id]['end_time'][-1]:
                        describe = command_dict['result']['uav_speed_record'][id]['describe'][-1]
                        self.uav_state_change[id][self.uav_state_class[id]] = {
                            describe if key == f'describe{self.uav_state_class[id]}' else key: value for key, value in
                            self.uav_state_change[id][self.uav_state_class[id]].items()}  # 更新阶段描述
                        self.uav_state_class[id] += 1

        if self.red_obs.simtime % self.sonar_update_times == 0:
            self.touch_buoys = []
            passive_touch_count = 0
            active_touch_count = 0
            self.red_obs.uav_field.passive_touch = []
            for uav_id in range(self.red_obs.uav_field.uav_nums):
                uav = self.red_obs.uav_field.uavs[uav_id]
                uav.buoy_touch_ids.clear()
                alive_buoys = list(filter(lambda x: not x.dead, uav.buoys))

                for id, buoy in enumerate(uav.buoys):
                    if not buoy.dead:
                        buoy.result_clear()  # 浮标探测信息初始化
                        buoy.sensor_detect(self.blue_obs, self.env_obs, thermocline=self.args.thermocline,
                                           sensor_data=sensor_data)  # 浮标探测信息
                        if buoy.touch:  # 探测到信息
                            uav.buoy_touch_ids.append(id)
                            self.touch_buoys.append(buoy)
                            if buoy.btype == 1:
                                active_touch_count += 1
                                if buoy.touch_time != self.red_obs.simtime or buoy.touch_time == 0:
                                    buoy.touch_time = self.red_obs.simtime
                                    target_pos_info = ""
                                    for i in range(len(buoy.target_feature)):
                                        target_pos_info += f'第{i + 1}目标的经度为{buoy.target_pos[i]["lon"]}、纬度为{buoy.target_pos[i]["lat"]}；'
                                    target_pos_info = target_pos_info[:-1]
                                    message = {"title": "主动声呐浮标发现可疑目标",
                                               "detail": f"主动声呐浮标发现{len(buoy.target_feature)}个可疑目标, " + target_pos_info,
                                               'time': self.red_obs.simtime}
                                    obs_message["red_message"]["uav_message"][uav_id].append(message) if message not in \
                                                                                                         obs_message[
                                                                                                             "red_message"][
                                                                                                             "uav_message"][
                                                                                                             uav_id] else None
                                buoy.touch_time += self.sonar_update_times

                            else:
                                passive_touch_count += 1
                                if buoy.touch_time != self.red_obs.simtime or buoy.touch_time == 0:
                                    buoy.touch_time = self.red_obs.simtime
                                    target_pos_info = ""
                                    for i in range(len(buoy.target_feature)):
                                        target_pos_info += f'第{i + 1}目标距离当前被动声呐浮标的方位角为{buoy.target_course[i]}、频率为{buoy.target_feature[i]["f"]}、信号声强级{buoy.target_feature[i]["p_recevied"]}；'
                                    target_pos_info = target_pos_info[:-1]
                                    message = {"title": "被动声呐浮标发现可疑目标",
                                               "detail": f"被动声呐浮标发现{len(buoy.target_feature)}个可疑目标, " + target_pos_info,
                                               'time': self.red_obs.simtime}
                                    obs_message["red_message"]["uav_message"][uav_id].append(message) if message not in \
                                                                                                         obs_message[
                                                                                                             "red_message"][
                                                                                                             "uav_message"][
                                                                                                             uav_id] else None
                                buoy.touch_time += self.sonar_update_times

        ####################### 多个被动声呐联合定位 ########################
        if self.red_obs.simtime % self.sonar_update_times == 0:
            multi_sonar_loc = passvive_sonar_combination(self.touch_buoys, self.blue_obs, self.env_obs)
            if multi_sonar_loc:
                if self.red_obs.combination_touch_time != self.red_obs.simtime or self.red_obs.combination_touch_time == 0:
                    self.red_obs.combination_touch_time = self.red_obs.simtime
                    target_pos_info = ""
                    for i in range(len(multi_sonar_loc)):
                        target_pos_info += f'第{i + 1}目标的纬度为{multi_sonar_loc[i]["lat"]}、经度为{multi_sonar_loc[i]["lon"]}、频率为{multi_sonar_loc[i]["f"]}；'
                    target_pos_info = target_pos_info[:-1]
                    message = {"title": "多个被动声呐浮标联合定位发现可疑目标",
                               "detail": f"多个被动声呐浮标联合发现{len(multi_sonar_loc)}个可疑目标, " + target_pos_info,
                               'time': self.red_obs.simtime}
                    obs_message["red_message"]["uav_message"][0].append(message) if message not in \
                                                                                    obs_message["red_message"][
                                                                                        "uav_message"][0] else None
                self.red_obs.combination_touch_time += self.sonar_update_times

                multi_sonar_loc.sort(key=lambda x: -x['f'])
                self.red_obs.virtual.lat = multi_sonar_loc[0]["lat"]
                self.red_obs.virtual.lon = multi_sonar_loc[0]["lon"]
                if self.red_obs.virtual_touch_time != self.red_obs.simtime or self.red_obs.virtual_touch_time == 0:
                    message = {"title": "红方发现可疑潜艇位置",
                               "detail": f"红方发现可疑潜艇位置，目标经度为{self.red_obs.virtual.lon}、目标纬度为{self.red_obs.virtual.lat}",
                               'time': self.red_obs.simtime}
                    obs_message["red_message"]["uav_message"][0].append(message) if message not in \
                                                                                    obs_message["red_message"][
                                                                                        "uav_message"][0] else None
                    # message = {"info": "红方获得蓝方潜艇应召点信息", "class": 'red', "type": 'buoy'}
                    # key_message.append(message) if message not in key_message else None
                self.red_obs.virtual_touch_time += self.sonar_update_times

        self.red_obs.buoy_message = command_dict["buoy_message"]
        if command_dict["task_message"]['uavs']:
            for i in range(self.red_obs.uav_field.uav_nums):
                self.red_obs.uav_field.uavs[i].task_message = command_dict["task_message"]['uavs'][i]

        if command_dict["task_message"]['usvs']:
            for i in range(self.red_obs.usv_field.usv_nums):
                self.red_obs.usv_field.usvs[i].task_message = command_dict["task_message"]['usvs'][i]

        # 传感器状态更新(除了声呐浮标)
        for id in range(self.red_obs.uav_field.uav_nums):
            if self.red_obs.uav_field.uavs[id].infrared.statu:
                self.red_obs.uav_field.uavs[id].infrared.result_clear()
                self.red_obs.uav_field.uavs[id].infrared.sensor_detect(self.red_obs.uav_field.uavs[id], self.blue_obs,
                                                                       self.env_obs, self.args, env_temp=290,
                                                                       sensor_data=sensor_data)  # 红外传感器更新
                if self.red_obs.uav_field.uavs[id].infrared.touch:
                    if self.red_obs.uav_field.uavs[id].infrared.touch_time != self.red_obs.simtime or \
                            self.red_obs.uav_field.uavs[id].infrared.touch_time == 0:
                        self.red_obs.uav_field.uavs[id].infrared.touch_time = self.red_obs.simtime
                        target_type = []
                        target_ref_pos = []
                        detail = ""
                        for i, info in enumerate(self.red_obs.uav_field.uavs[id].infrared.target_info):
                            target_type.append(info['type'])
                            target_ref_pos.append(info['ref_pos'])
                            detail += f"第{i + 1}个目标的类型为{info['type']}，参考位置的经度为{info['ref_pos']['lon']}、纬度{info['ref_pos']['lat']}；"
                        detail = detail[:-1]
                        message = {"title": "无人机红外传感器发现可疑目标",
                                   "detail": f"{id + 1}号无人机的红外传感器发现{len(target_type)}个可疑目标" + detail,
                                   'time': self.red_obs.simtime}
                        obs_message["red_message"]["uav_message"][id].append(message) if message not in \
                                                                                         obs_message["red_message"][
                                                                                             "uav_message"][
                                                                                             id] else None
                    self.red_obs.uav_field.uavs[id].infrared.touch_time += 1

            if self.red_obs.uav_field.uavs[id].magnetic.statu:
                self.red_obs.uav_field.uavs[id].magnetic.result_clear()
                self.red_obs.uav_field.uavs[id].magnetic.sensor_detect(self.red_obs.uav_field.uavs[id], self.blue_obs,
                                                                       self.env_obs, sensor_data=sensor_data)  # 磁探传感器更新
                if self.red_obs.uav_field.uavs[id].magnetic.touch:
                    if self.red_obs.uav_field.uavs[id].magnetic.touch_time != self.red_obs.simtime or \
                            self.red_obs.uav_field.uavs[id].magnetic.touch_time == 0:
                        self.red_obs.uav_field.uavs[id].magnetic.touch_time = self.red_obs.simtime
                        detail = ""
                        for i, info in enumerate(self.red_obs.uav_field.uavs[id].magnetic.target_pos):
                            detail += f"第{i + 1}个目标的经度为{info['lon']}, 纬度为{info['lat']}, 高度为{info['alt']}；"
                        detail = detail[:-1]
                        message = {"title": "无人机磁探传感器发现目标",
                                   "detail": f"{id + 1}号无人机的磁探传感器在经度{self.red_obs.uav_field.uavs[id].lon}、纬度{self.red_obs.uav_field.uavs[id].lat}处的磁异常数据为{self.red_obs.uav_field.uavs[id].magnetic.target_feature}，该磁探仪共发现{len(self.red_obs.uav_field.uavs[id].magnetic.target_pos)}个目标" + detail,
                                   'time': self.red_obs.simtime}

                        obs_message["red_message"]["uav_message"][id].append(message) if message not in \
                                                                                         obs_message["red_message"][
                                                                                             "uav_message"][
                                                                                             id] else None
                    self.red_obs.uav_field.uavs[id].magnetic.touch_time += 1

            self.red_obs.uav_field.uavs[id].battery.update_battery(self.red_obs.uav_field.uavs[id].vel * 1000 / 3600,
                                                                   200 * 1000 / 3600)  # 无人机电量更新

        for id in range(self.red_obs.usv_field.usv_nums):
            if self.red_obs.usv_field.usvs[id].sonar.statu and self.red_obs.simtime % self.sonar_update_times == 0:
                self.red_obs.usv_field.usvs[id].sonar.result_clear()
                self.red_obs.usv_field.usvs[id].sonar.sensor_detect(self.red_obs, self.blue_obs, self.env_obs, id,
                                                                    thermocline_height=self.args.thermocline,
                                                                    sensor_data=sensor_data)  # 无人艇拖曳声呐更新
                if self.red_obs.usv_field.usvs[id].sonar.touch:
                    self.usv_time_first_identified_sub = self.red_obs.simtime
                    if self.red_obs.usv_field.usvs[id].sonar.touch_time != self.red_obs.simtime or \
                            self.red_obs.usv_field.usvs[id].sonar.touch_time == 0:
                        detail = ""
                        for i, info in enumerate(self.red_obs.usv_field.usvs[id].sonar.target_pos):
                            detail += f"第{i + 1}个目标的经度为{info['lon']}、纬度{info['lat']}；"
                        detail = detail[:-1]
                        message = {"title": "无人艇发现目标",
                                   "detail": f"{id + 1}号无人艇的拖曳声呐共发现{len(self.red_obs.usv_field.usvs[id].sonar.target_pos)}个目标" + detail,
                                   'time': self.red_obs.simtime}
                        obs_message["red_message"]["usv_message"][id].append(message) if message not in \
                                                                                         obs_message["red_message"][
                                                                                             "usv_message"][
                                                                                             id] else None
                    self.red_obs.usv_field.usvs[id].sonar.touch_time += self.sonar_update_times

            self.red_obs.usv_field.usvs[id].battery.update_battery(
                self.red_obs.usv_field.usvs[id].vel * 1.852 * 1000 / 3600,
                22 * 1.852 * 1000 / 3600)  # 无人艇电量更新

        self.red_obs.simtime += 1
        self.blue_obs.simtime += 1

        # 15min 平均潜艇角度预测正确
        for id in range(self.blue_obs.submarine_nums):
            self.blue_sub_info[id].append(
                {"lat": self.blue_obs.submarines[id].lat, "lon": self.blue_obs.submarines[id].lon,
                 "course": self.blue_obs.submarines[id].course})

        # # 终止判断 如果红方成功跟踪15分钟，并能够实时上报QT位置，则判定红方胜利
        # plane_dones = [False for _ in range(self.blue_obs.submarine_nums)]
        #
        # for sub_id in range(self.blue_obs.submarine_nums):
        #     if self.red_obs.report.lat[sub_id] is not None:  # 存在上报点
        #         self.red_obs.start_track_time[sub_id][1].append(self.red_obs.simtime)
        #         self.red_obs.start_track_time[sub_id][0] += 1 #上报次数加一
        #
        #         self.red_obs.report_time_minute[sub_id].append(self.red_obs.simtime)
        #         minute_times = False # 每分钟是否上报了四次
        #         if self.red_obs.simtime - self.red_obs.report_time_minute[sub_id][0] <= 60:
        #             minute_times = True
        #         else:
        #             count = sum(1 for x in self.red_obs.report_time_minute[sub_id][::-1] if abs(self.red_obs.report_time_minute[sub_id][-1] - x) <= 60)
        #             if count >= 4:
        #                 minute_times = True
        #
        #         if minute_times:
        #             if len(self.blue_sub_info[0]) < 15 * 60:
        #                 true_course_avg = self.blue_sub_info[sub_id][-1]["course"]
        #             else:
        #                 g = geod.Inverse(self.blue_sub_info[sub_id][-15 * 60]['lat'], self.blue_sub_info[sub_id][-15 * 60]['lat'], self.blue_sub_info[sub_id][-1]['lat'], self.blue_sub_info[sub_id][-1]['lon'])
        #                 true_course_avg = g['azi1']
        #
        #             couse_error_avg = self.red_obs.report.course[sub_id] - true_course_avg
        #             couse_error_avg = couse_error_avg + 360 if couse_error_avg < 0 else couse_error_avg
        #             couse_error_avg = 360 - couse_error_avg if couse_error_avg > 180 else couse_error_avg
        #             g = geod.Inverse(self.red_obs.report.lat[sub_id], self.red_obs.report.lon[sub_id],
        #                              self.blue_obs.submarines[sub_id].lat,
        #                              self.blue_obs.submarines[sub_id].lon)
        #             couse_error = self.red_obs.report.course[sub_id] - self.blue_obs.submarines[sub_id].course
        #             couse_error = couse_error + 360 if couse_error < 0 else couse_error
        #             couse_error = 360 - couse_error if couse_error > 180 else couse_error
        #             vel_error = abs(self.red_obs.report.vel[sub_id] - self.blue_obs.submarines[sub_id].vel)
        #
        #             if g['s12'] < self.args.error_dis and vel_error < self.args.error_vel and (couse_error < self.args.error_course or couse_error_avg < self.args.error_course):  # 判断可疑位置的准确性
        #                 self.red_obs.report_success_time[sub_id][0] += 1
        #                 self.red_obs.report_success_time[sub_id][1].append(self.red_obs.simtime)
        #                 print("红方成功上报蓝方{}号潜艇{}分钟".format(sub_id, self.red_obs.report_success_time[sub_id][0] / 60))
        #                 if self.red_obs.report_success_time[sub_id][0] / 60 == 0:
        #                     message = {"title": "红方成功上报潜艇位置",  "detail": f"红方成功上报蓝方潜艇{self.red_obs.report_success_time[sub_id][0] / 60}分钟", 'time': self.red_obs.simtime}
        #                     obs_message["red_message"]["uav_message"][0].append(message) if message not in obs_message["red_message"]["uav_message"][0] else None
        #
        #
        #             if len(self.red_obs.start_track_time[sub_id][1]) >= 15 * 60:
        #                 if self.red_obs.report_success_time[sub_id][0] >= self.red_obs.start_track_time[sub_id][0] * 0.7:  # 15min中有70%时间上报成功则判断红方胜利
        #                     plane_dones[sub_id] = True
        #                     accuracy = round(self.red_obs.report_success_time[sub_id] * 100/self.red_obs.start_track_time[sub_id][0], 2)
        #                     message = {"title": "红方胜利", "detail": f"红方成功跟踪蓝方{self.red_obs.report_success_time[sub_id]/60}分钟，准确率为{accuracy}%", 'time': self.red_obs.simtime}
        #                     obs_message["red_message"]["uav_message"][0].append(message) if message not in obs_message["red_message"]["uav_message"][0] else None
        #                 else:
        #                     self.red_obs.start_track_time[sub_id][1] += 1
        #
        #         else:
        #             print('每分钟不能上报四次目标，重新计数')
        #             self.red_obs.clear_report_info()#每分钟不能上报4次目标，重新计数
        # 终止判断 如果红方成功跟踪15分钟，并能够实时上报QT位置，则判定红方胜利
        plane_dones = [False for _ in range(self.blue_obs.submarine_nums)]

        for sub_id in range(self.blue_obs.submarine_nums):
            if self.red_obs.report.lat[sub_id] is not None:  # 存在上报点
                if self.red_obs.start_track_time[sub_id] is None:  # 开始上报可疑位置
                    self.red_obs.start_track_time[sub_id] = self.red_obs.simtime
                self.red_obs.report_time_minute[sub_id].append(self.red_obs.simtime)
                minute_times = False  # 每分钟是否上报了四次
                if self.red_obs.simtime - self.red_obs.report_time_minute[sub_id][0] <= 60:
                    minute_times = True
                else:
                    count = sum(1 for x in self.red_obs.report_time_minute[sub_id][::-1] if
                                abs(self.red_obs.report_time_minute[sub_id][-1] - x) <= 60)
                    if count >= 4:
                        minute_times = True

                if minute_times:
                    if len(self.blue_sub_info[0]) < 15 * 60:
                        true_course_avg = self.blue_sub_info[sub_id][-1]["course"]
                    else:
                        g = geod.Inverse(self.blue_sub_info[sub_id][-15 * 60]['lat'],
                                         self.blue_sub_info[sub_id][-15 * 60]['lat'],
                                         self.blue_sub_info[sub_id][-1]['lat'],
                                         self.blue_sub_info[sub_id][-1]['lon'])
                        true_course_avg = g['azi1']

                    couse_error_avg = self.red_obs.report.course[sub_id] - true_course_avg
                    couse_error_avg = couse_error_avg + 360 if couse_error_avg < 0 else couse_error_avg
                    couse_error_avg = 360 - couse_error_avg if couse_error_avg > 180 else couse_error_avg
                    g = geod.Inverse(self.red_obs.report.lat[sub_id], self.red_obs.report.lon[sub_id],
                                     self.blue_obs.submarines[sub_id].lat,
                                     self.blue_obs.submarines[sub_id].lon)
                    couse_error = self.red_obs.report.course[sub_id] - self.blue_obs.submarines[sub_id].course
                    couse_error = couse_error + 360 if couse_error < 0 else couse_error
                    couse_error = 360 - couse_error if couse_error > 180 else couse_error
                    vel_error = abs(self.red_obs.report.vel[sub_id] - self.blue_obs.submarines[sub_id].vel)

                    if g['s12'] < self.args.error_dis and vel_error < self.args.error_vel and (
                            couse_error < self.args.error_course or couse_error_avg < self.args.error_course):  # 判断可疑位置的准确性
                        self.red_obs.report_success_time[sub_id] += 1
                        print("红方成功上报蓝方{}号潜艇{}分钟".format(sub_id,
                                                                      self.red_obs.report_success_time[sub_id] / 60))
                        if self.red_obs.report_success_time[sub_id] / 60 == 0:
                            message = {"title": "红方成功上报潜艇位置",
                                       "detail": f"红方成功上报蓝方潜艇{self.red_obs.report_success_time[sub_id] / 60}分钟",
                                       'time': self.red_obs.simtime}
                            obs_message["red_message"]["uav_message"][0].append(message) if message not in \
                                                                                            obs_message[
                                                                                                "red_message"][
                                                                                                "uav_message"][
                                                                                                0] else None

                        if (self.red_obs.simtime - self.red_obs.start_track_time[sub_id]) >= 15 * 60:
                            if self.red_obs.report_success_time[sub_id] >= (
                                    self.red_obs.simtime - self.red_obs.start_track_time[
                                sub_id]) * 0.7:  # 15min中有70%时间上报成功则判断红方胜利
                                plane_dones[sub_id] = True
                                accuracy = round(self.red_obs.report_success_time[sub_id] * 100 / (
                                        self.red_obs.simtime - self.red_obs.start_track_time[sub_id]), 2)
                                message = {"title": "红方胜利",
                                           "detail": f"红方成功跟踪蓝方{self.red_obs.report_success_time[sub_id] / 60}分钟，准确率为{accuracy}%",
                                           'time': self.red_obs.simtime}
                                obs_message["red_message"]["uav_message"][0].append(message) if message not in \
                                                                                                obs_message[
                                                                                                    "red_message"][
                                                                                                    "uav_message"][
                                                                                                    0] else None
                else:
                    self.red_obs.clear_report_info()  # 每分钟不能上报4次目标，重新计数
        if any(plane_dones):
            result = Result.RED_WIN
            done = True
        if self.red_obs.simtime >= self.args.Inference_time * 60 * 60:
            result = Result.BLUE_WIN
            done = True
            message = {"title": "蓝方胜利", "detail": f"蓝方胜利成功躲避红方{self.args.Inference_time}小时",
                       'time': self.red_obs.simtime}
            obs_message["blue_message"]["sub_message"][0].append(message) if message not in \
                                                                             obs_message["blue_message"]["sub_message"][
                                                                                 0] else None

        # sub_dones = []
        # for id in range(self.blue_obs.submarine_nums):
        #     if geod.Inverse(self.blue_obs.submarines[id].lat, self.blue_obs.submarines[id].lon,
        #                     self.blue_obs.task_point[id].lat,
        #                     self.blue_obs.task_point[id].lon)['s12'] < self.args.sub_target_dis:
        #         sub_dones.append(True)
        #     else:
        #         sub_dones.append(False)
        #
        #     # 只要有一艘潜艇到达了任务点 就算潜艇赢
        #     if any(sub_dones):
        #         result = Result.BLUE_WIN
        #         done = True

        ############# 环境要素状态更新 #############
        # 渔船位置更新
        for i in range(self.env_obs.fishing_boat_nums):
            self.env_obs.fishing_boats[i].lon, self.env_obs.fishing_boats[i].lat, self.env_obs.fishing_boats[
                i].angle, self.env_obs.fishing_boats[i].vel = FishingBoatControl(
                self.env_obs.fishing_boats[i]).fishing_boat_move()

        # 货轮位置更新
        for i in range(self.env_obs.cargo_ship_nums):
            self.env_obs.cargo_ships[i].lon, self.env_obs.cargo_ships[i].lat, self.env_obs.cargo_ships[
                i].angle = CargoShipControl(
                self.env_obs.cargo_ships[i]).cargo_ship_move()

        if result != 0:
            # 仿真结果统计
            # 声呐信息统计
            uav_passive_sonar_survival = 0
            uav_active_sonar_survival = 0
            buoy_nums = 0
            for id in range(self.red_obs.uav_field.uav_nums):
                uav = self.red_obs.uav_field.uavs[id]
                self.uav_sonar_buoy_num["passive"] += uav.buoy_passive_use
                self.uav_sonar_buoy_num["activate"] += uav.buoy_activate_use
                buoy_nums += (uav.buoy_passive_use + uav.buoy_activate_use)

                for buoy in uav.buoys:
                    if not buoy.dead:
                        if buoy.btype == 0:  # 被动声呐
                            uav_passive_sonar_survival += 1
                        else:
                            uav_active_sonar_survival += 1

            self.uav_passive_sonar_survival_rate = uav_passive_sonar_survival / buoy_nums  # 被动声呐存活率
            self.uav_active_sonar_survival_rate = uav_active_sonar_survival / buoy_nums  # 主动声呐存活率
            self.uav_total_duration_call_point = command_dict['result']['uav_total_duration_call_point']
            self.uav_time_first_identified_sub = command_dict['result']['uav_time_first_identified_sub']
            self.usv_total_duration_call_point = command_dict['result']['usv_total_duration_call_point']

            # 蓝方
            self.sub_initial_exposure_time = min(self.uav_time_first_identified_sub, self.usv_time_first_identified_sub)
            self.result_statistics["blue_result"] = {
                "sub_type": self.sub_type,  #
                "sub_nums": self.blue_obs.submarine_nums,  #
                "sub_jammer_deployed_num": self.sub_jammer_deployed_num,  #
                "sub_bait_deployed_num": self.sub_bait_deployed_num,  #
                "sub_total_navigation_mileage": self.sub_total_navigation_mileage,  #
                "sub_total_duration": self.blue_obs.simtime,  #
                "sub_velocity_list": self.sub_velocity_list,  #
                "sub_action_list": self.sub_action_list,
                "sub_state_list": self.sub_state_list,
                "sub_low_speed_sailing_duration": self.sub_low_speed_sailing_duration,  #
                "sub_high_speed_sailing_duration": self.sub_high_speed_sailing_duration,  #
                "sub_goal_completion_rate": self.sub_goal_completion_rate,  #
                "sub_initial_exposure_time": self.sub_initial_exposure_time  #
            }
            self.result_statistics["red_result"] = {
                "call_point": [{"lat": self.red_obs.call_point[id].lat, "lon": self.red_obs.call_point[id].lon} for id
                               in range(self.blue_obs.submarine_nums)],  #

                "uav_type": self.uav_type,  #
                "uav_nums": self.red_obs.uav_field.uav_nums,  #
                "uav_total_duration_call_point": self.uav_total_duration_call_point,
                "uav_time_first_identified_sub": self.uav_time_first_identified_sub,
                "uav_total_navigation_mileage": self.uav_total_navigation_mileage,  #
                "uav_sonar_buoy_num": self.uav_sonar_buoy_num,
                "uav_passive_sonar_survival_rate": self.uav_passive_sonar_survival_rate,
                "uav_active_sonar_survival_rate": self.uav_active_sonar_survival_rate,
                "uav_target_recognition_accuracy": self.uav_target_recognition_accuracy,
                'uav_vel_change': self.uav_state_change,
                "usv_type": self.usv_type,  #
                "usv_nums": self.red_obs.usv_field.usv_nums,  #
                "usv_total_navigation_mileage": self.usv_total_navigation_mileage,  #
                "usv_total_duration_call_point": self.usv_total_duration_call_point,
                "usv_time_first_identified_sub": self.usv_time_first_identified_sub,  #
            }
            # 环境
            self.result_statistics["env_result"] = {
                "env_fishing_boat_nums": self.env_obs.fishing_boat_nums,  #
                "env_cargo_ship_nums": self.env_obs.cargo_ship_nums,  #
                "env_vortex_nums": self.env_obs.vortex_nums,  #
                "env_wreck_nums": self.env_obs.wreck_nums,  #
                "env_fish_nums": self.env_obs.fish_nums,  #
            }

        return {"red_obs": self.red_obs,
                "blue_obs": self.blue_obs,
                "env_obs": self.env_obs,
                "obs_message": obs_message,
                "key_message": key_message,
                "result_statistics": self.result_statistics}, result


class EnvTest:
    def __init__(self, args, task_id=None, episode_i=None):
        self.args = args
        self.env = SouthCall(self.args, task_id, episode_i)

    def reset(self):
        return self.env.reset()

    def step(self, cmds):
        return self.env.step(cmds)


"""
决策生成
"""


class Decision:
    def __init__(self):
        self.plane_agent = RedAgent()
        self.sub_agent = SubDecision()

    def make_decision(self, raw_obs, task_id=None, episode_i=None):
        commands = {
            "red_cmds": [],
            "blue_cmds": []
        }
        blue_obs = raw_obs["blue_obs"]
        red_obs = raw_obs["red_obs"]
        env_obs = raw_obs["env_obs"]

        # g = geod.Inverse(blue_obs.lat, blue_obs.lon, red_obs.lat, red_obs.lon)
        # s_p_dis = g["s12"]
        # g = geod.Inverse(blue_obs.lat, blue_obs.lon, blue_obs.task_point.lat, blue_obs.task_point.lon)
        # s_t_dis = g["s12"]
        # print(blue_obs.simtime, red_obs.simtime, red_obs.lat, red_obs.lon, s_p_dis, s_t_dis)
        sub_cmds, commands["blue_message"], commands['blue_key_message'] = self.sub_agent.make_decision(blue_obs)
        commands["blue_cmds"] = sub_cmds

        if red_obs.simtime >= 0:
            red_cmds = self.plane_agent.make_decision(red_obs, blue_obs, task_id, episode_i)
            commands["red_cmds"], commands["task_message"], commands["buoy_message"], commands['result'] = red_cmds
        else:
            commands["task_message"] = "待命"
            commands["buoy_message"] = ""
        return commands
