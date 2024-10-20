# """
# 态势解析类
# """
# """
# 态势类
# """
# import argparse
# import base64
# from io import BytesIO
# from typing import List
# import matplotlib.pyplot as plt
# import numpy as np
# from geographiclib.geodesic import Geodesic
#
# geod = Geodesic.WGS84
# import random
#
#
#
#
#
# ######################## 红方态势 ########################
#
# ########################蓝方态势 #######################
#
# class EnvInfo:
#     def __init__(self):
#         self.center = Position()
#         self.sea_condition = random.randint(1, 5)  # 海况
#         self.sys_time = random.randint(0, 24)  # 系统时间
#
#
# class Red_Sonar:
#     def __init__(self):
#         """agent_type: uav, usv;
#         sonar_type: 0--被动， 1--主动"""
#
#         self.sonar_type = 0
#         self.lat = None
#         self.lon = None
#         self.height = None
#         self.statu = 0
#         self.result = []
#         self.course = None
#         self.img = None
#         self.find_times = 0
#
#     def update(self, red_obs, blue_obs, sonar_type, sensor_img=False):
#         self.img = None
#         self.lat = None
#         self.lon = None
#         self.sonar_type = sonar_type
#         # 针对单潜艇目标
#
#         for obs in blue_obs.submarines:
#             self.img, self.result, self.course, self.lat, self.lon = get_red_sonar_img(red_obs.lat, red_obs.lon,
#                                                                                        obs.lat,
#                                                                                        obs.lon,
#                                                                                        sonar_type,
#                                                                                        sensor_img=sensor_img)
#         # if self.result:
#         #     self.course = self.course + np.random.uniform() * 5
#         #     if sonar_type:
#         #         self.lat = self.lat + np.random.uniform() * 0.002
#         #         self.lon = self.lon + np.random.uniform() * 0.002
#         #     if self.find_times % 5 == 0:
#         #         self.img, _, _, _, _ = get_red_sonar_img(red_obs.lat, red_obs.lon, obs.lat, obs.lon,
#         #                                                  self.agent_type, sonar_type, sensor_img=True)
#         #     self.find_times += 1
#         # else:
#         #     self.find_times = 0
#
#
# class Blue_Sonar:
#     def __init__(self, sonar_type):
#         """sonar_type: 0--被动， 1--主动"""
#         self.sonar_type = sonar_type
#         self.lat = None
#         self.lon = None
#         self.height = None
#         self.statu = 1
#         self.result = None
#         self.course = None
#         self.img = None
#
#     def update(self, red_obs, env_obs, blue_obs):
#         error = [np.random.uniform() * 0.002, np.random.uniform() * 0.002, np.random.uniform() * 5]
#         self.lat, self.lon, self.course, self.result = get_blue_sonar_img(blue_obs, red_obs, env_obs, self.sonar_type,
#                                                                           error)
#
#
# class Sonar_Wave:
#     def __init__(self, wave_type, wave_height, wave_direction, wave_speed, wave_time):
#         self.wave_type = wave_type
#
#
# class Buoy:
#     def __init__(self, btype, lat, lon, channel, height, start_time):
#         self.btype = btype
#         self.lat = lat
#         self.lon = lon
#         self.channel = channel
#         self.height = height
#         self.touch_flag = False
#         self.start_time = start_time
#         self.dead = True if np.random.rand() < 0.1 else False
#         self.course = 0
#         self.target_lat = None
#         self.target_lon = None
#         self.find_times = 0
#         self.img = None
#         self.never_touch = True
#         self.bait_attract_times = 0
#         self.target_type = None  # 目标识别
#         if self.btype == 62 or self.btype == 63:
#             self.range = 1200
#         elif self.btype == 67 or self.btype == 68:
#             self.range = 3000
#
#         self.jammer_effect = False#是否被干扰器影响
#         self.bait_flag = False#是否被声诱饵touch
#
#
#     def result_clear(self):
#         self.touch_flag = False
#         self.course = None
#         self.target_lat = None
#         self.target_lon = None
#         self.img = None
#         self.jammer_effect = False
#         self.bait_flag = False
#
#         # self.wave = None
#
#     def __repr__(self):
#         return str(self.btype)
#         # return str(self.btype) + '-' + str(self.channel) + '-' + str(self.touch)
#         # return str(self.touch) + '-' + str(self.dead)
#
#     def update(self, red_obs, blue_obs, sensor_img=False):
#         self.img = None
#         # self.lat = None
#         # self.lon = None
#
#         # 针对单潜艇目标
#         for obs in blue_obs.submarines:
#             self.img, _, self.course, self.target_lat, self.target_lon = get_buoy_img(red_obs.lat,
#                                                                                       red_obs.lon,
#                                                                                       obs.lat,
#                                                                                       obs.lon,
#                                                                                       self.channel,
#                                                                                       sensor_img=sensor_img)
#         # if self.channel == 70:
#         #     if self.touch_flag:
#         #         self.course = self.course + np.random.uniform() * 5
#         #         self.target_lat = self.target_lat + np.random.uniform() * 0.002
#         #         self.target_lon = self.target_lon + np.random.uniform() * 0.002
#         #         if self.find_times % 5 == 0:
#         #             self.img, _, _, _, _ = get_buoy_img(red_obs.lat, red_obs.lon, obs.lat, obs.lon, self.channel,
#         #                                                 sensor_img=True)
#         #         self.find_times += 1
#         #     else:
#         #         self.find_times = 0
#         #
#         # else:
#         #     if self.touch_flag:
#         #         self.course = self.course + np.random.uniform() * 5
#         #         if self.find_times % 5 == 0:
#         #             self.img, _, _, _, _ = get_buoy_img(red_obs.lat, red_obs.lon, obs.lat, obs.lon, self.channel,
#         #                                                 sensor_img=True)
#         #         self.find_times += 1
#         #     else:
#         #         self.find_times = 0
#
#
# # 磁探
# class Magnetic:
#     def __init__(self):
#         self.statu = 0  # 是否开启
#         self.valid = 0  # 是否有效
#         self.lat = None
#         self.lon = None
#         self.result = 0
#
#         # plt.imshow(np.zeros((100, 100)))
#         # save_file = BytesIO()
#         # plt.savefig(save_file, format="png")
#         # self.img = base64.b64encode(save_file.getvalue()).decode('utf8')
#         self.img = None
#         self.find_times = 0
#
#     def update(self, red_obs, blue_obs, sensor_img=False):
#         #   针对多目标潜艇
#         # self.lat = []
#         # self.lon = []
#         # sub_params = []  # 针对多个潜艇的情况
#         # for obs in blue_obs.submarines:
#         #     sub_params.append(
#         #         {'sub_lat': obs.lat, 'sub_lon': obs.lon, 'sub_alt': obs.height})
#         # self.img, self.result = MagneticModel().cal_magnetic(sub_params, red_obs.lat, red_obs.lon, red_obs.height)
#         # id = 0
#         # print('11', self.result)
#         # for result in self.result:
#         #     if result:
#         #         self.lat.append(blue_obs.submarines[id].lat + np.random.uniform() * 0.005)
#         #         self.lon.append(blue_obs.submarines[id].lon + np.random.uniform() * 0.005)
#         #     else:
#         #         self.lat.append(None)
#         #         self.lon.append(None)
#         #     id += 1
#
#         # 针对单目标潜艇
#         sub_params = []
#         self.img = None
#         self.lat = None
#         self.lon = None
#         for obs in blue_obs.submarines:
#             sub_params.append(
#                 {'sub_lat': obs.lat, 'sub_lon': obs.lon, 'sub_alt': obs.height})
#             _, self.result = MagneticModel().cal_magnetic(sub_params, red_obs.lat, red_obs.lon, red_obs.height,
#                                                           sensor_img=sensor_img)
#             self.result = self.result[0]
#             if self.result:
#                 self.lat = obs.lat + np.random.uniform() * 0.002
#                 self.lon = obs.lon + np.random.uniform() * 0.002
#                 if self.find_times % 5 == 0:
#                     self.img, _ = MagneticModel().cal_magnetic(sub_params, red_obs.lat, red_obs.lon, red_obs.height,
#                                                                sensor_img=True)
#                 self.find_times += 1
#             else:
#                 self.find_times = 0
#
#
# # 雷达
# class Radar:
#     def __init__(self):
#         self.statu = 0
#         self.valid = 0
#         self.result = 0
#         self.lat = None
#         self.lon = None
#         # self.result = np.random.random((20, 20)).tolist()
#         self.wave = None
#         self.course = None  # 目标的朝向，以当前飞机为中心点，单位度
#         self.find_times = 0
#         self.img = None
#
#     def update(self, red_obs, blue_obs, sensor_img=False):
#         self.img = None
#         self.lat = None
#         self.lon = None
#
#         for obs in blue_obs.submarines:
#             _, self.result, self.course = RadarModel().gen_radar_img(red_obs.lat, red_obs.lon, obs.lat, obs.lon,
#                                                                      sensor_img=sensor_img)
#             if self.result:
#                 self.course = self.course + np.random.uniform() * 5
#                 self.lat = obs.lat + np.random.uniform() * 0.002
#                 self.lon = obs.lon + np.random.uniform() * 0.002
#                 if self.find_times % 5 == 0:
#                     self.img, _, _ = RadarModel().gen_radar_img(red_obs.lat, red_obs.lon, obs.lat,
#                                                                 obs.lon,
#                                                                 sensor_img=True)
#                 self.find_times += 1
#             else:
#                 self.find_times = 0
#
#
# # 光电
# class Photo:
#     def __init__(self):
#         self.statu = 0
#         self.valid = 0
#         # self.img = np.random.random((20, 20)).tolist()
#         # self.img = np.zeros((100, 100)).tolist()
#         # plt.imshow(np.zeros((100, 100)))
#         # save_file = BytesIO()
#         # plt.savefig(save_file, format="png")
#         # self.img = base64.b64encode(save_file.getvalue()).decode('utf8')
#         self.img = None
#         self.result = 0
#         self.lat = None
#         self.lon = None
#         self.find_times = 0
#
#     def update(self, red_obs, blue_obs, phi=0, psi=0, theta=0, sub_psi=60, sensor_img=False):
#         """plane_lat, plane_lon:：飞机的经纬度海拔
#            sub_lat, sub_lon： 潜艇的经纬度海拔
#            phi, psi, theta： 飞机的滚转角、偏航角、俯仰角 # 偏航角,朝着正北为0度，-180 - 180度
#            sub_psi： 潜艇的偏航角"""
#         #   针对多目标潜艇
#         # self.lat = []
#         # self.lon = []
#         # plane_params = {'plane_lat': red_obs.lat, 'plane_lon': red_obs.lon, 'plane_height': red_obs.height,
#         #                 'phi': phi, 'psi': red_obs.course, 'theta': theta}
#         # sub_params = []  # 针对多个潜艇的情况
#         # for obs in blue_obs.submarines:
#         #     sub_params.append({'sub_lat': obs.lat, 'sub_lon': obs.lon, 'sub_height': obs.height,
#         #                        'sub_psi': obs.course})  ## 偏航角,朝着正北为0度，-180 - 180度
#         # self.img, self.result = infrared().run(plane_params, sub_params)
#         # id = 0
#         # for result in self.result:
#         #     if result:
#         #         self.lat.append(blue_obs.submarines[id].lat + np.random.uniform() * 0.005)
#         #         self.lon.append(blue_obs.submarines[id].lon + np.random.uniform() * 0.005)
#         #     else:
#         #         self.lat.append(None)
#         #         self.lon.append(None)
#         #     id += 1
#
#         # 针对单目标潜艇
#         sub_params = []
#         plane_params = {'plane_lat': red_obs.lat, 'plane_lon': red_obs.lon, 'plane_height': red_obs.height, 'phi': phi,
#                         'psi': red_obs.course, 'theta': theta}
#         for obs in blue_obs.submarines:
#             sub_params.append({'sub_lat': obs.lat, 'sub_lon': obs.lon, 'sub_height': obs.height,
#                                'sub_psi': obs.course})
#             _, self.result = infrared().run(plane_params, sub_params, sensor_img=sensor_img)
#             self.result = self.result[0]
#             if self.result:
#                 self.lat = obs.lat + np.random.uniform() * 0.002
#                 self.lon = obs.lon + np.random.uniform() * 0.002
#                 if not self.img or self.find_times % 3 == 0:
#                     self.img, _ = infrared().run(plane_params, sub_params, sensor_img=True)
#                 self.find_times += 1
#             else:
#                 self.img = None
#                 self.find_times = 0
#                 self.lat = None
#                 self.lon = None
#
#
# # 潜艇干扰器
# class Jammer:
# class Jammer:
#     def __init__(self, lat, lon, height, start_time):
#         self.lat = lat
#         self.lon = lon
#         self.height = height
#         self.lure_flag = False  # 是否干扰到对手
#         self.start_time = start_time
#         self.alive_time = 35 * 60  # 剩余生活时间
#         self.dead = False if self.alive_time > 0 else True
#         self.course = 0
#         self.wave = None
#
#     def update(self, now_time, red_obs):
#         self.alive_time = max(35 * 60 - (now_time - self.start_time), 0)  # 剩余生活时间
#         self.dead = False if self.alive_time > 0 else True
#         if not self.dead:
#             for uav_id in range(red_obs.uav_field.uav_nums):
#                 for id, buoy in enumerate(red_obs.uav_field.uavs[uav_id].buoys):
#                     if buoy.dead:
#                         pass
#                     else:
#                         jammer_flag, jammer_level = Jammer_Model().gen_jammer(self.lat, self.lon, buoy.lat, buoy.lon)
#                         if jammer_flag:
#                             buoy.jammer_effect = True
#                         else:
#                             buoy.jammer_effect = False
#         return red_obs
#
#
# # 潜艇声诱饵
# class acoustic_bait:
#     def __init__(self, lat, lon, height, velocity, course, transfer, start_time):
#         self.lat = lat
#         self.lon = lon
#         self.last_lat = lat
#         self.last_lon = lon
#         self.height = height
#         self.velocity = velocity
#         self.course = course
#         self.transfer = transfer
#         self.attract = False  # 是否引诱到对手
#         self.start_time = start_time
#         self.alive_time = 40 * 60  # 剩余工作时间
#         self.dead = False if self.alive_time > 0 else True
#
#     def update(self, now_time, red_obs):
#         self.alive_time = max(40 * 60 - (now_time - self.start_time), 0)  # 剩余生活时间
#         self.dead = False if self.alive_time > 0 else True
#         if not self.dead:
#             # 更新变换后的速度和方向
#             for trans in self.transfer:
#                 if now_time - self.start_time == trans["trans_time"]:
#                     self.velocity = trans["trans_velocity"]
#                     self.course = trans["trans_course"]
#                     break
#             # 获取声诱饵的位置信息
#             s12 = self.velocity * 1.83 * 1000 / 3600
#             g = geod.Direct(lat1=self.last_lat, lon1=self.last_lon, azi1=self.course, s12=s12)
#             self.lat = g["lat2"]
#             self.lon = g["lon2"]
#             self.last_lat = self.lat
#             self.last_lon = self.lon
#
#             for uav_id in range(red_obs.uav_field.uav_nums):
#                 for id, buoy in enumerate(red_obs.uav_field.uavs[uav_id].buoys):
#                     if buoy.dead:
#                         pass
#                     else:
#                         img, attract = acoustic_bait_Model().gen_bait(self.lat, self.lon, buoy.lat, buoy.lon)
#                         if attract:
#                             print('声诱饵吸引到声呐浮标, id:', id)
#                             buoy.bait_flag = True
#                             # buoy.target_lat = self.lat + np.random.uniform() * 0.002
#                             # buoy.target_lon = self.lon + np.random.uniform() * 0.002
#                             if buoy.bait_attract_times % 3 == 0:#避免图片一直显示
#                                 buoy.img, _ = acoustic_bait_Model().gen_bait(self.lat, self.lon, buoy.lat, buoy.lon,
#                                                                              sensor_img=True)
#                                 buoy.bait_attract_times += 1
#                         else:
#                             buoy.bait_attract_times = 0
#                         # red_obs.uav_field.uavs[uav_id].buoys[id] = buoy
#
#         return red_obs
#
#
# # 潜艇潜望镜
# class Periscope:
#     def __init__(self):
#         self.detect_range = 10_000  # 单位m
#         self.long = 10  # 镜筒长度，单位m
#         self.statu = 0
#         self.result = []
#
#
# class Position:
#     def __init__(self, lat=None, lon=None, alt=None):
#         self.lat = alt
#         self.lon = alt
#         self.alt = alt
#         self.target_type = None
#
#
# class Unit:
#     def __init__(self):
#         self.lat = 0
#         self.lon = 0
#         self.height = 0
#         self.course = 0
#         self.vel = 0
#
#
# class Battery:
#     def __init__(self, en_time_max, battery_max, battery_ratio, en_dis_max):
#         """
#         en_time_max: 续航时间最大 单位：分钟
#         en_time: 初始化续航时间 单位：分钟
#         battery_max: 油箱最大值（单位：kg）
#         battery: 油箱初始化大小，百分比
#         en_dis_max: 最大续航距离 单位：km
#         en_dis: 初始化续航距离 单位：km
#         battery_use:使用的油量/电量
#         """
#
#         self.en_time_max = en_time_max  # 续航时间最大 单位：分钟
#         self.battery_max = battery_max  # 油箱最大值（单位：kg）或者是 电池最大值（单位：Mwh）
#         self.battery_ratio = battery_ratio  # 油箱初始化大小，百分比
#         self.en_dis_max = en_dis_max  # 最大续航距离 单位：km
#         self.en_time = round(self.en_time_max * self.battery_ratio / 100, 2)  # 初始化续航时间 单位：分钟
#         self.en_dis = round(self.en_dis_max * self.battery_ratio / 100, 2)  # 初始化续航距离 单位：km
#         self.battery_use = 0
#
#     def update_battery(self, vel, vel_min):
#         """计算剩余电量/油量"""
#         a = self.battery_max / (self.en_time_max * 60 * vel_min ** 2)
#         battery = self.battery_ratio * self.battery_max / 100 - a * vel ** 2
#         self.battery_ratio = max(round(battery * 100 / self.battery_max, 2), 0)
#         self.en_time = max(round(self.en_time_max * battery / self.battery_max, 2), 0)  # 单位分
#         self.en_dis = max(round(self.en_dis_max * battery / self.battery_max, 2), 0)
#         self.battery_use += a * vel ** 2
#
#
# class UAV(Unit):
#     def __init__(self, args):
#         super(UAV, self).__init__()
#         # self.args = args
#
#         self.buoys: List[Buoy] = []
#         self.buoy_touch_ids = []
#
#         self.buoy_62_nums = 10  # 被动声呐浮标
#         self.buoy_63_nums = 10
#         self.buoy_67_nums = 10
#         self.buoy_68_nums = 10  # 主动声呐浮标
#         self.buoy_70_nums = 2
#
#         self.buoy_62_max_nums = 10
#         self.buoy_63_max_nums = 10
#         self.buoy_67_max_nums = 10
#         self.buoy_68_max_nums = 10
#
#         self.buoy_62_use = 0
#         self.buoy_63_use = 0
#         self.buoy_67_use = 0
#         self.buoy_68_use = 0
#
#         self.buoy_detect_range = {
#             "62": 1200,
#             "63": 1200,
#             "67": 3000,
#             "68": 3000,
#             "70": 2000
#         }
#         self.magnetic = Magnetic()
#         self.radar = Radar()
#         self.photo = Photo()
#
#         # self.virtual = Position()
#         # self.report = Position()
#         # self.report = [Position() for _ in range(args.submarine_nums)]
#         # self.call_point = Position()
#         # self.entry_point = Position()
#         self.result = 0
#
#         # 无人机续航信息 -- 彩虹-4无人机
#         self.battery = Battery(en_time_max=38 * 60, battery_max=165, battery_ratio=100, en_dis_max=5000)
#
#         self.task_message = ""
#         self.mileage = 0  # 无人机行驶里程
#         self.never_buoy = True  # 是否从来没有投放浮标，发送message使用
#         self.buoy_array_over = False  # 是否投放完声呐浮标阵列
#
#     def update_params(self):
#         self.last_lat = self.lat
#         self.last_lon = self.lon
#
#
# class UAVField:
#     def __init__(self, args):
#         self.args = args
#         self.uav_nums = self.args.uav_nums
#         self.uavs = self.gen_uavs()
#
#     def gen_uavs(self):
#         uavs = []
#         for id in range(self.uav_nums):
#             uavs.append(UAV(self.args))
#         return uavs
#
#
# class USV(Unit):
#     def __init__(self, args):
#         # self.args = args
#         self.sonar = Red_Sonar()  # 拖拽声呐
#         # self.photo = Photo()  # 光电传感器
#         # 无人艇续航信息 -- 瞭望者Ⅱ
#         self.battery = Battery(en_time_max=10 * 60, battery_max=45, battery_ratio=100, en_dis_max=574.12)
#
#         self.task_message = ""
#         self.lat = None
#         self.lon = None
#         self.height = 0
#         self.phi = 0
#         self.theta = 0
#         self.psi = 0
#         self.vel = 0
#         self.mileage = 0  # 无人艇行驶里程
#         self.course = 0
#
#         # 容量
#         self.buoy_62_nums = 100
#         self.buoy_68_nums = 24
#
#     def update_params(self):
#         self.last_lat = self.lat
#         self.last_lon = self.lon
#         self.last_height = self.height
#         self.last_phi = self.phi  # 滚转角
#         self.last_theta = self.theta  # 俯仰角
#         self.last_psi = self.psi  # 偏航角
#         # self.last_vel = self.vel
#
#     def MotionSim_update(self, target_lat, target_lon, target_height, target_u):
#         self.lat, self.lon, self.height, self.phi, self.theta, self.psi, self.vel = USVMotionSim(self.last_lat,
#                                                                                                  self.last_lon,
#                                                                                                  self.last_height,
#                                                                                                  self.last_phi,
#                                                                                                  self.last_theta,
#                                                                                                  self.last_psi).run(
#             target_lat, target_lon, target_height, target_u)
#         self.mileage += geod.Inverse(self.last_lat, self.last_lon, self.lat, self.lon)['s12']
#         self.update_params()
#
#
# class USVField:
#     def __init__(self, args):
#         self.args = args
#         self.usv_nums = self.args.usv_nums
#         self.usvs = self.gen_usvs()
#
#     def gen_usvs(self):
#         usvs = []
#         for id in range(self.usv_nums):
#             usvs.append(USV(self.args))
#         return usvs
#
#
# class RedGlobalObservation:
#     def __init__(self, args):
#         # self.args = args
#         self.simtime = 0
#         self.uav_field = UAVField(args)
#         self.usv_field = USVField(args)
#         self.entry_point = Position()
#         self.call_point = Position()
#         self.multi_sensor_img = None
#         self.task_message = ""
#         self.buoy_message = ""
#         self.virtual = Position()
#         self.report = Position()
#         self.last_report = Position()
#         self.track_time = 0
#         self.start_track_time = None
#         self.vir_history = []
#         self.vir_history_index = -1
#         self.last_vir = [[], []]
#         self.passive_62_touch = []
#         self.passive_63_touch = []
#         self.passive_67_touch = []
#         self.active_68_touch = []
#         self.vir_his_pos = []#记录历史可疑位置
#         self.report_plan = False
#         self.sub_pos = []  # 持续追踪过程中记录潜艇的位置
#
#
# """
# 蓝方全局态势
# """
#
#
# class BlueGlobalObservation:
#     def __init__(self, args):
#         self.simtime = 0
#         # self.args = args
#         self.submarine_nums = args.submarine_nums
#         # self.usv_nums = self.args.usv_nums
#
#         # self.usvs = [USV() for _ in range(self.usv_nums)]
#         self.submarines = [Submarine(args) for _ in range(self.submarine_nums)]
#         self.task_point = Position()
#
#
# """
# 潜艇态势类 （实体类） 可以探测到主动声呐浮标
# """
#
#
# class Submarine(Unit):
#     def __init__(self, args):
#         super(Submarine, self).__init__()
#         # self.args = args
#         self.sonar = Blue_Sonar(sonar_type=0)
#         # self.active_sonar = Sonar(agent_type="submarine", sonar_type=1)
#         self.jammer_nums = 2  # 干扰器数量
#         self.jammers = []
#         self.bait_nums = 2  # 声诱饵数量
#         self.bait = []
#
#         self.snorkel = False  # 通气管状态，为了充电
#         self.periscope = Periscope()
#
#         # 潜艇续航信息
#         self.battery = Battery(en_time_max=4.5 * 24 * 60, battery_max=70, battery_ratio=82, en_dis_max=500)
#
#         self.task_message = ""
#         self.lat = None
#         self.lon = None
#         self.height = None
#         self.phi = 0
#         self.theta = 0
#         self.psi = 0
#         self.vel = 0
#         self.finish_degree = 0  # 潜艇任务完成度，百分比
#         self.mileage = 0  # 潜艇行驶里程
#
#     def update_params(self):
#         self.last_lat = self.lat
#         self.last_lon = self.lon
#         self.last_height = self.height
#         self.last_phi = self.phi  # 滚转角
#         self.last_theta = self.theta  # 俯仰角
#         self.last_psi = self.psi  # 偏航角
#         self.last_vel = self.vel
#
#     def MotionSim_update(self, target_lat, target_lon, target_height,
#                          target_u):
#         self.lat, self.lon, self.height, self.phi, self.theta, self.psi, self.vel = SubMotionSim(self.last_lat,
#                                                                                                  self.last_lon,
#                                                                                                  self.last_height,
#                                                                                                  self.last_phi,
#                                                                                                  self.last_theta,
#                                                                                                  self.last_psi).run(
#             target_lat, target_lon, target_height, target_u)
#         self.mileage += geod.Inverse(self.last_lat, self.last_lon, self.lat, self.lon)['s12']
#         self.update_params()
#
#
#
# if __name__ == '__main__':
#     a = np.random.random((5, 5))
