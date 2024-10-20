import time

from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84
from ..sensor import magnetic as Magnetic_model
from ..sensor.Infrared import infrared
from typing import List
import numpy as np
from ..sensor import sonar_class as Sonar
from ..sensor import blue_sonar as drag_Sonar
import torch.nn as nn

class Position:
    def __init__(self, lat=None, lon=None, height=None):
        self.lat = lat
        self.lon = lon
        self.height = height
        self.target_type = None

class REPORT:
    def __init__(self, args):
        self.sub_nums = args.submarine_nums
        self.lat = [None for _ in range(self.sub_nums)]
        self.lon = [None for _ in range(self.sub_nums)]
        self.height = [None for _ in range(self.sub_nums)]
        self.course = [None for _ in range(self.sub_nums)]
        self.vel = [None for _ in range(self.sub_nums)]
        self.target_type = [None for _ in range(self.sub_nums)]
        self.report_id = [{"uav":[], "usv":[]} for i in range(self.sub_nums)]
        self.report_time = [None for _ in range(self.sub_nums)]
    def clear_info(self):
        """上报信息清空"""
        self.lat = [None for _ in range(self.sub_nums)]
        self.lon = [None for _ in range(self.sub_nums)]
        self.height = [None for _ in range(self.sub_nums)]
        self.course = [None for _ in range(self.sub_nums)]
        self.vel = [None for _ in range(self.sub_nums)]
        self.target_type = [None for _ in range(self.sub_nums)]
        self.report_id = [{"uav": [], "usv": []} for i in range(self.sub_nums)]
        self.report_time = [None for _ in range(self.sub_nums)]


def thermocline_detect(sonar_height, thermocline, target_height):
    if sonar_height is None:
        return True
    if (abs(sonar_height) <= abs(thermocline) and abs(target_height) <= abs(thermocline)) or (
            abs(sonar_height) >= abs(thermocline) and abs(target_height) >= abs(thermocline)):
        return True
    else:
        return False

def passvive_sonar_combination(buoy_list, blue_obs, env_obs, thermocline=-90):
    """多被动声呐联合定位"""
    env = passvive_sonar_env(blue_obs, env_obs, buoy_list, thermocline=thermocline)
    passive_sonars = []
    for buoy in buoy_list:
        sonar = Sonar.passive_sonar(sonar_lat=buoy.lat, sonar_lon=buoy.lon, sonar_alt=buoy.height, v=0, env=env, name="被动声呐浮标")
        passive_sonars.append(sonar)
    sonar_combination = Sonar.passive_sonar_combination(passive_sonars)
    res = sonar_combination.analyse_location_type(plt_show=True)
    if not res:
        return None
    else:
        return res
def passvive_sonar_env(blue_obs, env_obs, buoy_list, thermocline=90):
    """生成多个被动声呐联合探测需要的声源信息"""
    target = []  # 需求生成多个声源，包括潜艇、渔船
    random_min = -10
    random_max = 80
    buoy_touch_times = {}
    jammer_range = 4000
    Jammers = []
    buoy_touch_loc_nums = 2
    Baits =[]

    for i in range(blue_obs.submarine_nums):
        for jammer in blue_obs.submarines[i].jammers:
            if not jammer.dead:
                Jammers.append(jammer)
        for bait in blue_obs.submarines[i].bait:
            if not bait.dead:
                Baits.append(bait)

    for buoy in buoy_list:
        for i in range(blue_obs.submarine_nums):
            buoy_touch_times.update({f'submarine{i}': [0, False]}) if all(f'submarine{i}' not in d for d in buoy_touch_times) else None
            detect_range = np.clip(0.5 * blue_obs.submarines[i].vel - 1, 1, 4) * 1851 + np.random.uniform(
                    random_min, random_max)
            if geod.Inverse(buoy.lat, buoy.lon, blue_obs.submarines[i].lat, blue_obs.submarines[i].lon)[
                's12'] <= detect_range and thermocline_detect(buoy.height, thermocline, blue_obs.submarines[i].height) and buoy_touch_times[f'submarine{i}'][0]<buoy_touch_loc_nums:
                buoy_touch_times[f'submarine{i}'][0] += 1
                for jammer in Jammers:  # 潜艇四周是否有干扰器
                    if geod.Inverse(jammer.lat, jammer.lon, blue_obs.submarines[i].lat, blue_obs.submarines[i].lon)['s12'] <= jammer_range:
                        buoy_touch_times[f'submarine{i}'][0] -= 1
                        break
            if buoy_touch_times[f'submarine{i}'][0] >= buoy_touch_loc_nums and not buoy_touch_times[f'submarine{i}'][1]:
                target.append(Sonar.Target(sub_lat=blue_obs.submarines[i].lat, sub_lon=blue_obs.submarines[i].lon,
                                           sub_alt=blue_obs.submarines[i].height, v=-blue_obs.submarines[i].vel,
                                           f=1000 + np.random.uniform(40, 80),
                                           P=50, target_density=10,
                                           sub_name="潜艇"))
                buoy_touch_times[f'submarine{i}'][1] = True

        for i in range(len(Baits)):
            buoy_touch_times.update({f'bait{i}': [0, False]}) if all(f'bait{i}' not in d for d in buoy_touch_times) else None
            detect_range = np.clip(0.5 * Baits[i].vel - 1, 1, 4) * 1851 + np.random.uniform(random_min, random_max)
            if geod.Inverse(buoy.lat, buoy.lon, Baits[i].lat, Baits[i].lon)['s12'] <= detect_range and thermocline_detect(buoy.height, thermocline, Baits[i].height) and buoy_touch_times[f'bait{i}'][0]<buoy_touch_loc_nums:
                buoy_touch_times[f'bait{i}'][0] += 1
                for jammer in Jammers:  # 潜艇四周是否有干扰器
                    if geod.Inverse(jammer.lat, jammer.lon, Baits[i].lat, Baits[i].lon)['s12'] <= jammer_range:
                        buoy_touch_times[f'bait{i}'][0] -= 1
                        break
            if buoy_touch_times[f'bait{i}'][0] >= buoy_touch_loc_nums and not buoy_touch_times[f'bait{i}'][1]:
                target.append(
                            Sonar.Target(sub_lat=Baits[i].lat, sub_lon=Baits[i].lon,
                                         sub_alt=Baits[i].height, v=-Baits[i].vel,
                                         f=1000 + np.random.uniform(40, 80),
                                         P=50, target_density=10,
                                         sub_name="声诱饵"))
                buoy_touch_times[f'bait{i}'][1] = True


        for i in range(env_obs.fishing_boat_nums):  # 渔船信息
            buoy_touch_times.update({f'fishing_boat{i}': [0, False]}) if all(f'fishing_boat{i}' not in d for d in buoy_touch_times) else None
            detect_range = np.random.uniform(0.8, 1) * 1851 + np.random.uniform(random_min, random_max)
            if geod.Inverse(buoy.lat, buoy.lon, env_obs.fishing_boats[i].lat, env_obs.fishing_boats[i].lon)[
                's12'] <= detect_range and thermocline_detect(buoy.height, thermocline, 0) and buoy_touch_times[f'fishing_boat{i}'][0]<buoy_touch_loc_nums:
                buoy_touch_times[f'fishing_boat{i}'][0] += 1
                for jammer in Jammers:  # 潜艇四周是否有干扰器
                    if geod.Inverse(jammer.lat, jammer.lon, env_obs.fishing_boats[i].lat,
                                    env_obs.fishing_boats[i].lon)['s12'] <= jammer_range:
                        buoy_touch_times[f'fishing_boat{i}'][0] -= 1
                        break
            if buoy_touch_times[f'fishing_boat{i}'][0] >= buoy_touch_loc_nums and not buoy_touch_times[f'fishing_boat{i}'][1]:
                target.append(Sonar.Target(sub_lat=env_obs.fishing_boats[i].lat, sub_lon=env_obs.fishing_boats[i].lon,
                                           sub_alt=0, v=-10, f=200 + np.random.uniform(10, 20), P=1, target_density=5,
                                           sub_name="渔船"))
                buoy_touch_times[f'fishing_boat{i}'][1] = True

        for i in range(env_obs.cargo_ship_nums):  # 货轮信息
            buoy_touch_times.update({f'cargo_ship{i}': [0, False]}) if all(
                f'cargo_ship{i}' not in d for d in buoy_touch_times) else None
            detect_range = np.random.uniform(1, 1.2) * 1851 + np.random.uniform(random_min, random_max)
            if geod.Inverse(buoy.lat, buoy.lon, env_obs.cargo_ships[i].lat, env_obs.cargo_ships[i].lon)[
                's12'] <= detect_range and thermocline_detect(buoy.height, thermocline, 0) and buoy_touch_times[f'cargo_ship{i}'][0]<buoy_touch_loc_nums:
                buoy_touch_times[f'cargo_ship{i}'][0] += 1
                for jammer in Jammers:  # 潜艇四周是否有干扰器
                    if geod.Inverse(jammer.lat, jammer.lon, env_obs.cargo_ships[i].lat,
                                    env_obs.cargo_ships[i].lon)['s12'] <= jammer_range:
                        buoy_touch_times[f'cargo_ship{i}'][0] -= 1
                        break
            if buoy_touch_times[f'cargo_ship{i}'][0] >= buoy_touch_loc_nums and not buoy_touch_times[f'cargo_ship{i}'][1]:
                target.append(Sonar.Target(sub_lat=env_obs.cargo_ships[i].lat, sub_lon=env_obs.cargo_ships[i].lon,
                               sub_alt=0, v=-17, f=300 + np.random.uniform(20, 30), P=5, target_density=10,
                               sub_name="货轮"))
                buoy_touch_times[f'cargo_ship{i}'][1] = True

        for i in range(env_obs.fish_nums):  # 鱼群信息
            buoy_touch_times.update({f'fish{i}': [0, False]}) if all(
                f'fish{i}' not in d for d in buoy_touch_times) else None
            detect_range = np.random.uniform(0.6, 0.8) * 1851 + np.random.uniform(random_min, random_max)
            if geod.Inverse(buoy.lat, buoy.lon, env_obs.fishs[i].lat, env_obs.fishs[i].lon)[
                's12'] <= detect_range and thermocline_detect(buoy.height, thermocline, env_obs.fishs[i].height) and buoy_touch_times[f'fish{i}'][0]<2:
                buoy_touch_times[f'fish{i}'][0] += 1
                for jammer in Jammers:  # 潜艇四周是否有干扰器
                    if geod.Inverse(jammer.lat, jammer.lon, env_obs.fishs[i].lat,
                                    env_obs.fishs[i].lon)['s12'] <= jammer_range:
                        buoy_touch_times[f'fish{i}'][0] -= 1
                        break
            if buoy_touch_times[f'fish{i}'][0] >= 2 and not buoy_touch_times[f'fish{i}'][1]:
                target.append(Sonar.Target(sub_lat=env_obs.fishs[i].lat, sub_lon=env_obs.fishs[i].lon,
                                   sub_alt=env_obs.fishs[i].height, v=-7, f=100 + np.random.uniform(8, 15), P=3,
                                   target_density=10,
                                   sub_name="鱼群"))
                buoy_touch_times[f'fish{i}'][1] = True

    env = Sonar.Environment(target)
    return env

def sonar_env(blue_obs, env_obs, lat, lon, vel, name, thermocline=90, height=None):
    """生成声呐探测需要的声源信息"""
    target = []  # 需求生成多个声源，包括潜艇、渔船
    detect_range = 1851
    random_min = -10
    random_max = 80
    jammer_range = 4000
    noise = False
    Jammers = []
    Baits = []
    for i in range(blue_obs.submarine_nums):
        for jammer in blue_obs.submarines[i].jammers:
            if not jammer.dead:
                Jammers.append(jammer)
        for bait in blue_obs.submarines[i].bait:
            if not bait.dead:
                Baits.append(bait)

    for sub in blue_obs.submarines:
        pop = False
        for jammer in sub.jammers:  # 四周是否有干扰器
            if geod.Inverse(jammer.lat, jammer.lon, lat, lon)['s12'] <= jammer_range:
                noise = True
                pop = True
                break
        if pop:
            break

    if name == '拖曳声呐':
        detect_range = np.clip(-0.25 * vel + 4, 1, 4) * 1851 + np.random.uniform(random_min, random_max)
    if name == '主动声呐浮标':
        detect_range = 2 * 1851 + np.random.uniform(random_min, random_max)
    for i in range(blue_obs.submarine_nums):
        if name == "被动声呐浮标":
            detect_range = np.clip(0.5 * blue_obs.submarines[i].vel - 1, 1, 4) * 1851 + np.random.uniform(
                random_min, random_max)
        if geod.Inverse(lat, lon, blue_obs.submarines[i].lat, blue_obs.submarines[i].lon)[
            's12'] <= detect_range and thermocline_detect(height, thermocline, blue_obs.submarines[i].height):
            target.append(Sonar.Target(sub_lat=blue_obs.submarines[i].lat, sub_lon=blue_obs.submarines[i].lon,
                                       sub_alt=blue_obs.submarines[i].height, v=-blue_obs.submarines[i].vel,
                                       f=1000 + np.random.uniform(40, 80),
                                       P=50, target_density=10,
                                       sub_name="潜艇"))
            for jammer in Jammers:  # 潜艇四周是否有干扰器
                if geod.Inverse(jammer.lat, jammer.lon, blue_obs.submarines[i].lat, blue_obs.submarines[i].lon)['s12'] <= jammer_range:
                    target.pop()
                    break


    for j in range(len(Baits)):
        if name == "被动声呐浮标":
            detect_range = np.clip(0.5 * Baits[j].vel - 1, 1,
                                   4) * 1851 + np.random.uniform(random_min, random_max)
        if geod.Inverse(lat, lon, Baits[j].lat, Baits[j].lon)[
            's12'] <= detect_range and thermocline_detect(height, thermocline, Baits[j].height):
            target.append(
                Sonar.Target(sub_lat=Baits[j].lat, sub_lon=Baits[j].lon,
                             sub_alt=Baits[j].height, v=-Baits[j].vel,
                             f=1000 + np.random.uniform(40, 80),
                             P=50, target_density=10,
                             sub_name="声诱饵"))
            for jammer in Jammers:  # 潜艇四周是否有干扰器
                if geod.Inverse(jammer.lat, jammer.lon, Baits[j].lat,
                                Baits[j].lon)['s12'] <= jammer_range:
                    target.pop()
                    break

    for j in range(env_obs.fishing_boat_nums):  # 渔船信息
        if name == "被动声呐浮标":
            detect_range = np.random.uniform(0.8, 1) * 1851 + np.random.uniform(random_min, random_max)
        if geod.Inverse(lat, lon, env_obs.fishing_boats[j].lat, env_obs.fishing_boats[j].lon)[
            's12'] <= detect_range and thermocline_detect(height, thermocline, 0):
            target.append(Sonar.Target(sub_lat=env_obs.fishing_boats[j].lat, sub_lon=env_obs.fishing_boats[j].lon,
                                       sub_alt=0, v=-10, f=200 + np.random.uniform(10, 20), P=1, target_density=5,
                                       sub_name="渔船"))
            for jammer in Jammers:
                if geod.Inverse(jammer.lat, jammer.lon, env_obs.fishing_boats[j].lat,
                                env_obs.fishing_boats[j].lon)['s12'] <= jammer_range:
                    target.pop()
                    break

    for j in range(env_obs.cargo_ship_nums):  # 货轮信息
        if name == "被动声呐浮标":
            detect_range = np.random.uniform(1, 1.2) * 1851 + np.random.uniform(random_min, random_max)
        if geod.Inverse(lat, lon, env_obs.cargo_ships[j].lat, env_obs.cargo_ships[j].lon)[
            's12'] <= detect_range and thermocline_detect(height, thermocline, 0):
            target.append(Sonar.Target(sub_lat=env_obs.cargo_ships[j].lat, sub_lon=env_obs.cargo_ships[j].lon,
                                       sub_alt=0, v=-17, f=300 + np.random.uniform(20, 30), P=5, target_density=10,
                                       sub_name="货轮"))
            for jammer in Jammers:  # 潜艇四周是否有干扰器
                if geod.Inverse(jammer.lat, jammer.lon, env_obs.cargo_ships[j].lat,
                                env_obs.cargo_ships[j].lon)['s12'] <= jammer_range:
                    target.pop()
                    break

    for j in range(env_obs.fish_nums):  # 鱼群信息
        if name == "被动声呐浮标":
            detect_range = np.random.uniform(0.6, 0.8) * 1851 + np.random.uniform(random_min, random_max)
        if geod.Inverse(lat, lon, env_obs.fishs[j].lat, env_obs.fishs[j].lon)[
            's12'] <= detect_range and thermocline_detect(height, thermocline, env_obs.fishs[j].height):
            target.append(Sonar.Target(sub_lat=env_obs.fishs[j].lat, sub_lon=env_obs.fishs[j].lon,
                                       sub_alt=env_obs.fishs[j].height, v=-7, f=100 + np.random.uniform(8, 15), P=3,
                                       target_density=10,
                                       sub_name="鱼群"))
            for jammer in Jammers:  # 潜艇四周是否有干扰器
                if geod.Inverse(jammer.lat, jammer.lon, env_obs.fishs[j].lat,
                                env_obs.fishs[j].lon)['s12'] <= jammer_range:
                    target.pop()
                    break
    env = Sonar.Environment(target)
    return env, noise

def magnetic_env(blue_obs, env_obs):
    objects_list = []
    for i in range(blue_obs.submarine_nums):
        objects_list.append(Magnetic_model.MagneticObject(lat=blue_obs.submarines[i].lat, lon=blue_obs.submarines[i].lon,
                                                         alt=blue_obs.submarines[i].height, M_m=80))
    for i in range(env_obs.fishing_boat_nums):  # 渔船信息
        objects_list.append(
            Magnetic_model.MagneticObject(lat=env_obs.fishing_boats[i].lat, lon=env_obs.fishing_boats[i].lon,
                                          alt=0, M_m=100))
    for i in range(env_obs.cargo_ship_nums):  # 货轮信息
        objects_list.append(
            Magnetic_model.MagneticObject(lat=env_obs.cargo_ships[i].lat, lon=env_obs.cargo_ships[i].lon,
                                          alt=0, M_m=100))

    return objects_list

class Red_Sonar:
    def __init__(self, hydrophone_num=35, hydrophone_dis=20):
        """usv的拖曳声呐:只允许被动探测"""
        self.statu = 0  # 0:未激活，1:激活
        self.touch = False  # 是否探测到可疑目标
        #{"theta": 信号方位（measured in radians）, "f": 信号频率（hz),  "p_recevied": 接收信号声强级}
        self.target_feature = []  # 每个水听器对每个目标的特征信息， 比如两个目标时结果{'sonar0': [{'theta': xx, 'f': xx, 'p_recevied': xx}, {'theta': xx, 'f': xx, 'p_recevied': xx}], 'sonar1': ..., 'sonar2': ...}，theta表示目标到当前水听器的角度，已经转化成-180到180的形式
        self.target_data = None# 每个水听器接收到的信号{'sonar0': {'x': array(), 'y': array()}, 'sonar1': .., 'sonar2': ..}
        self.target_pos = None#多个目标的位置信息，比如两个目标时结果：[{'lat': xx, 'lon': xx}, {'lat': xx, 'lon': xx}],一个目标时结果：[{'lat': xx, 'lon': xx}
        self.hydrophone_num = hydrophone_num  # 拖曳声呐探测水听器阵元个数
        self.hydrophone_dis = hydrophone_dis  # 阵元间隔
        self.theta_rope = None  # 绳缆与海平面夹角
        self.rope_len = 800  # 绳缆长度
        self.theta_hydrophone = None  # 拖曳阵与绳缆夹角
        self.hydrophone_pos = None  # 记录水听器的经纬度 比如有三个水听器组成的拖曳浮标，结果：'sonar0': {'lat': xx, 'lon': xx, 'height': xx}, 'sonar1': {'lat': xx, 'lon': xx, 'height': xx}, 'sonar2': {'lat': xx, 'lon': xx, 'height': xx}}
        self.noise = False #判断是否四周存在干扰器
        self.open_time = 0
        self.touch_time = 0

    def sonar_control(self, statu, theta_rope=10, rope_len=800, theta_hydrophone=20):
        self.statu = statu
        self.theta_rope = theta_rope
        self.rope_len = rope_len
        self.theta_hydrophone = theta_hydrophone

    def result_clear(self):
        self.touch = False
        self.target_feature = []
        self.target_data = None
        self.target_pos = None
        self.noise = False
        self.hydrophone_pos = None

    def sensor_detect(self, red_obs, blue_obs, env_obs, id, thermocline_height=-90, sensor_data=False):
        """sensor_data:是否返回声呐探测数据---用于前端画图
        theta_rope:绳缆与海平面夹角
        以D型缆阵为例，其主要技术参数包括拖缆直径、长度和重量，以及拖曳阵的直径、长度和重量。
        具体来说，拖缆直径可能为9.5mm，长度达到800m，重量为205kg；
        而拖曳阵直径可能是82.5mm（也有报道为89mm），长度为75m，重量为640kg。
        JT_lat,JT_lon,JT_alt: 舰艇位置
        v_lon,v_lat: 舰艇速度分量
        theta_rope:绳缆与海平面夹角
        theta_hydrophone：拖曳阵与绳缆夹角
        theta_rope+theta_hydrophone<90
        thermocline_height:跃变层
        """
        import time
        a = time.time()
        env, self.noise = sonar_env(blue_obs, env_obs, red_obs.usv_field.usvs[id].lat, red_obs.usv_field.usvs[id].lon,
                        red_obs.usv_field.usvs[id].vel, name="拖曳声呐")
        Drag_array_sonar = drag_Sonar.Passive_drag_array_sonar(env, rope_len=self.rope_len,
                                                          hydrophone_num=self.hydrophone_num,
                                                          hydrophone_dis=self.hydrophone_dis)  # rope_len:绳缆长度、hydrophone_num：阵元个数、hydrophone_dis：阵元间隔
        res_Drag_array = Drag_array_sonar.analyse_location(JT_lat=red_obs.usv_field.usvs[id].lat,
                                                           JT_lon=red_obs.usv_field.usvs[id].lon, JT_alt=0, v_lon=10,
                                                           v_lat=-30, theta_rope=self.theta_rope,
                                                           theta_hydrophone=self.theta_hydrophone,
                                                           thermocline_height=thermocline_height, plt_show=sensor_data)
        if not res_Drag_array:
            self.touch = False
            self.target_feature = []
            self.target_data = None
            self.target_pos = None
            self.hydrophone_pos = None
        else:
            self.touch = True
            if sensor_data:
                self.target_data = res_Drag_array['sonar_info']['data']
                self.target_data['x'] = self.target_data['x'].tolist() #numpy转为list
                self.target_data['y'] = self.target_data['y'].tolist()
            else:
                self.target_data = None
            self.hydrophone_pos = res_Drag_array['sonar_info']['feature']['pos']
            self.target_feature = res_Drag_array['sonar_info']['feature']['feature_info']
            for i in range(len(self.target_feature)):
                for j in range(len(self.target_feature[f'sonar{i}'])):
                    self.target_feature[f'sonar{i}'][j]['theta'] = Sonar.theta_to_angle(
                        self.target_feature[f'sonar{i}'][j]['theta'])  # 方位角变化为geod适用形式
            self.target_pos = res_Drag_array['target_pos']
            for info in self.target_pos:
                info.pop('f')


class Buoy:
    def __init__(self, btype, lat, lon, channel, height, start_time):
        self.btype = btype  # 0--被动， 1--主动
        self.lat = lat
        self.lon = lon
        self.channel = channel
        self.height = height
        self.touch = False
        self.start_time = start_time
        self.dead = False
        self.target_course = None #被动声呐：目标的方位，比如两个目标[xx, xx]，主动声呐为None
        self.target_pos = None#被动声呐：None, 主动声呐[{"lat": 0, "lon": 0, "height": 0}， "lat": 0, "lon": 0, "height": 0}]
        self.find_times = 0
        self.target_data = None#被动声呐和主动声呐：声呐的接收信号{'x': array(), 'y': array()}
        self.never_touch = True
        self.target_feature = None#被动声呐：加入存在两个目标，[{'theta': xx, 'f': xx, 'p_recevied': xx}, {'theta': xx, 'f': xx, 'p_recevied': xx}]。   主动声呐：[{'f': xx, 'p': xx, 'bias': xx, 'r': xx}, {'f': xx, 'p': xx, 'bias': xx, 'r':xx}] -- {"f": 回波频率f(hz),  "p": 回波声强级, "bias":回波时间(s), "pos":(lat,lon,alt), "r":目标距离（m)}
        self.noise = False  # 判断是否四周存在干扰器
        self.touch_time = 0

    def result_clear(self):
        self.touch = False
        self.target_feature = None
        self.target_data = None
        self.target_pos = None
        self.target_course = None
        self.noise = False

    def __repr__(self):
        return str(self.btype)

    def sensor_detect(self, blue_obs, env_obs, thermocline=-90, sensor_data=False):
        """thermocline:跃变层距离"""
        self.noise = False
        if self.btype == 0:  # 被动声呐
            env, self.noise = sonar_env(blue_obs, env_obs, self.lat, self.lon, 0, name="被动声呐浮标", thermocline=thermocline,
                            height=self.height)
            sonar = Sonar.passive_sonar(sonar_lat=self.lat, sonar_lon=self.lon, sonar_alt=self.height, v=0, env=env,
                                        name="被动声呐浮标")
            res = sonar.passive_detect(plt_show=sensor_data)  # 单个被动声呐
            if not res:
                self.touch = False
                self.target_feature = None
                self.target_data = None
                self.target_pos = None
                self.course = None
            else:
                self.touch = True
                if sensor_data:
                    self.target_data = {"x": res[1].tolist(), "y": res[2].tolist()}
                    self.target_feature = res[0]
                else:
                    self.target_data = None
                    self.target_feature = res
                for i in range(len(self.target_feature)):
                    self.target_feature[i]['theta'] = Sonar.theta_to_angle(
                        self.target_feature[i]['theta'])  # 方位角变化为geod适用形式
                self.target_pos = None
                self.target_course = [info['theta'] for info in self.target_feature]
        else:  # 主动声呐
            env, self.noise = sonar_env(blue_obs, env_obs, self.lat, self.lon, 0, name="主动声呐浮标", thermocline=thermocline,
                            height=self.height)
            sonar = Sonar.active_sonar(sonar_lat=self.lat, sonar_lon=self.lon, sonar_alt=self.height,
                                       dir=[75.06, -21.11, 214.31], v=0, env=env, f=2000, P=1, angle=10, name="主动声呐浮标")
            res = sonar.acive_detect(plt_show=sensor_data)
            if not res:
                self.touch = False
                self.target_feature = None
                self.target_data = None
                self.target_pos = None
                self.target_course = None
            else:
                self.touch = True
                if sensor_data:
                    self.target_data = {"x": res[1].tolist(), "y": res[2].tolist()}
                    self.target_feature = res[0]
                else:
                    self.target_data = None
                    self.target_feature = res
                self.target_pos = [{"lat": 0, "lon": 0, "height": 0} for i in range(len(self.target_feature))]
                for i in range(len(self.target_feature)):
                    self.target_pos[i]["lat"] = self.target_feature[i]['pos'][0]
                    self.target_pos[i]["lon"] = self.target_feature[i]['pos'][1]
                    self.target_pos[i]["height"] = self.target_feature[i]['pos'][2]
                for info in self.target_feature:
                    info.pop('pos')
                self.target_course = None


# 磁探
class Magnetic:
    def __init__(self):
        self.statu = 0  # 0:未激活，1:激活
        self.touch = False  # 是否探测到可疑目标
        self.target_feature = 0 #磁异常数值， 单个数值
        self.target_pos = None #[{"lat": xx, "lon": xx, "height": xx}, {"lat": xx, "lon": xx, "height": xx}]
        self.open_time = 0
        self.touch_time = 0

    def result_clear(self):
        self.touch = False
        self.target_feature = None
        self.target_pos = None

    def sensor_detect(self, red_obs, blue_obs, env_obs, sensor_data=False):
        if self.statu:
            objects_list = magnetic_env(blue_obs, env_obs)
            magnetic = Magnetic_model.Magnetic(lat=red_obs.lat, lon=red_obs.lon, alt=red_obs.height, R_m=800)
            env = Magnetic_model.Environment(objects_list, magnetic=magnetic, number=0)
            detect_value, detect_object_pos = env.detect_intensity()
            if len(detect_object_pos) == 0:
                self.touch = False
                self.target_feature = 0
                self.target_pos = None
            else:
                self.target_feature = detect_value
                self.touch = True
                self.target_pos = []
                for pos in detect_object_pos:
                    self.target_pos.append({"lat": pos[0], "lon": pos[1], "height": pos[2]})

# 红外
class Infrared:
    def __init__(self):
        self.statu = 0  # 0:未激活，1:激活
        self.touch = False  # 是否探测到可疑目标
        self.target_feature = None  # 图片形式
        self.target_info = None  # 探测结果信息，{'type': '潜艇', 'ref_pos': {'lat': xx 'lon': xx}, 'find': True}, {'type': '渔船', 'ref_pos': {}, 'find': True}, {'type': '渔船', 'ref_pos': {}, 'find': True}]
        self.target_pos = None #探测到潜艇的位置信息，{'lat': xx, 'lon': xx}
        self.infrared_detect = infrared()
        self.open_time = 0
        self.touch_time = 0

    def result_clear(self):
        self.touch = False
        self.target_feature = None
        self.target_info = None
        self.target_pos = None

    def sensor_detect(self, red_obs, blue_obs, env_obs, args, env_temp=290, sensor_data=False):

        if self.statu:
            env_params = {'temp': env_temp, 'sea_state': args.Env_params_Sea_state}
            sub_params = []
            fisherboat_params = []
            plane_param = {'lat': red_obs.lat, 'lon': red_obs.lon, 'height': red_obs.height,
                           'phi': red_obs.phi,
                           'psi': red_obs.psi, 'theta': red_obs.theta}
            for obs in blue_obs.submarines:
                sub_params.append(
                    {'lat': obs.lat, 'lon': obs.lon, 'height': obs.height, 'vel': obs.vel * 0.5144,
                     'psi': obs.course})  # 输入速度单位m/s

            for i in range(len(env_obs.fishing_boats)):
                fisherboat_params.append({'lat': env_obs.fishing_boats[i].lat, 'lon': env_obs.fishing_boats[i].lon,
                                          'length': env_obs.fishing_boats[i].length,
                                          "weight": env_obs.fishing_boats[i].width,
                                          "vel": env_obs.fishing_boats[i].vel * 0.5144,
                                          'psi': env_obs.fishing_boats[i].angle})  # 渔船船长和宽度的单位为m， 输入速度单位m/s

            result = self.infrared_detect.detect(plane_param, sub_params, fisherboat_params, env_params,
                                                 plt_show=sensor_data)
            if not result[0]:
                self.touch = False
                self.target_feature = None
                self.target_info = None
                self.target_pos = None
            else:
                self.touch = True
                if sensor_data:
                    self.target_feature = result[1].tolist()
                else:
                    self.target_feature = None
                self.target_info = result[0]
                self.target_pos = None
                for info in self.target_info:
                    if info['type'] == '潜艇':
                        self.target_pos = info['ref_pos']



class Unit:
    def __init__(self):
        self.lat = 0
        self.lon = 0
        self.height = 0
        self.course = 0
        self.vel = 0


class Battery:
    def __init__(self, en_time_max, battery_max, battery_ratio, en_dis_max):
        """
        en_time_max: 续航时间最大 单位：分钟
        en_time: 初始化续航时间 单位：分钟
        battery_max: 油箱最大值（单位：kg）
        battery: 油箱初始化大小，百分比
        en_dis_max: 最大续航距离 单位：km
        en_dis: 初始化续航距离 单位：km
        battery_use:使用的油量/电量
        """

        self.en_time_max = en_time_max  # 续航时间最大 单位：分钟
        self.battery_max = battery_max  # 油箱最大值（单位：kg）或者是 电池最大值（单位：Mwh）
        self.battery_ratio = battery_ratio  # 油箱初始化大小，百分比
        self.en_dis_max = en_dis_max  # 最大续航距离 单位：km
        self.en_time = round(self.en_time_max * self.battery_ratio / 100, 2)  # 初始化续航时间 单位：分钟
        self.en_dis = round(self.en_dis_max * self.battery_ratio / 100, 2)  # 初始化续航距离 单位：km
        self.battery_use = 0

    def update_battery(self, vel, vel_min, snorkel=False):
        """计算剩余电量/油量"""
        if not snorkel:
            a = self.battery_max / (self.en_time_max * 60 * vel_min ** 2)
            battery = self.battery_ratio * self.battery_max / 100 - a * vel ** 2
            self.battery_ratio = max(round(battery * 100 / self.battery_max, 2), 0)
            self.en_time = max(round(self.en_time_max * battery / self.battery_max, 2), 0)  # 单位分
            self.en_dis = max(round(self.en_dis_max * battery / self.battery_max, 2), 0)
            self.battery_use += a * vel ** 2
        else:
            battery = max(self.battery_ratio * self.battery_max / 100 + np.random.uniform(0.5, 0.8), self.battery_max)
            self.battery_ratio = max(round(battery * 100 / self.battery_max, 2), 0)
            self.en_time = max(round(self.en_time_max * battery / self.battery_max, 2), 0)  # 单位分
            self.en_dis = max(round(self.en_dis_max * battery / self.battery_max, 2), 0)


class UAV(Unit):
    def __init__(self, id, buoy_passive_nums, buoy_active_nums, args=None):
        super(UAV, self).__init__()
        # self.args = args

        self.id = id
        self.buoys: List[Buoy] = []
        self.buoy_touch_ids = []

        self.buoy_passive_nums = buoy_passive_nums  # 被动声呐浮标
        self.buoy_activate_nums = buoy_active_nums# 主动声呐浮标

        self.buoy_passive_max_nums = buoy_passive_nums
        self.buoy_activate_max_nums = buoy_active_nums


        self.buoy_passive_use = 0
        self.buoy_activate_use = 0


        self.magnetic = Magnetic()

        self.infrared = Infrared()

        self.result = 0

        # 无人机续航信息
        self.battery = Battery(en_time_max=38 * 60, battery_max=165, battery_ratio=100, en_dis_max=5000)

        self.task_message = ""
        self.mileage = 0  # 无人机行驶里程
        self.never_buoy = True  # 是否从来没有投放浮标，发送message使用
        self.buoy_array_over = False  # 是否投放完声呐浮标阵列
        self.psi = 0
        self.phi = 0
        self.theta = 0
        self.type = None

    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon


class UAVField():
    def __init__(self, args):
        # self.args = args
        self.uav_nums = args.uav_nums
        self.buoy_passive_nums = args.buoy_nums['passive']
        self.buoy_active_nums = args.buoy_nums['active']
        self.uavs = self.gen_uavs()

    def gen_uavs(self):
        uavs = []
        for id in range(self.uav_nums):
            uavs.append(UAV(id, self.buoy_passive_nums, self.buoy_active_nums))
        return uavs



class USV(Unit):
    def __init__(self, id, Usv_params_Sonar):
        # self.args = args
        self.id = id
        if Usv_params_Sonar:
            self.sonar = Red_Sonar()  # 拖拽声呐
        # self.photo = Photo()  # 光电传感器
        # 无人艇续航信息 -- 瞭望者Ⅱ
        self.battery = Battery(en_time_max=10 * 60, battery_max=45, battery_ratio=100, en_dis_max=574.12)

        self.task_message = ""
        self.lat = None
        self.lon = None
        self.height = 0
        self.phi = 0
        self.theta = 0
        self.psi = 0
        self.vel = 0
        self.mileage = 0  # 无人艇行驶里程
        self.course = 0
        self.type = None


    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon
        self.last_height = self.height
        self.last_phi = self.phi  # 滚转角
        self.last_theta = self.theta  # 俯仰角
        self.last_psi = self.psi  # 偏航角
        # self.last_vel = self.vel


class USVField:
    def __init__(self, args):
        # self.args = args
        self.usv_nums = args.usv_nums
        self.Usv_params_Sonar = args.Usv_params_Sonar
        self.usvs = self.gen_usvs()

    def gen_usvs(self):
        usvs = []
        for id in range(self.usv_nums):
            usvs.append(USV(id, self.Usv_params_Sonar))
        return usvs


class PLANE(Unit):
    def __init__(self, id, args=None):
        self.id = id
        self.lat = None
        self.lon = None
        self.height = 0
        self.vel = 0
        self.mileage = 0
        self.course = 0
        self.type = None

    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon
        self.last_height = self.height


class PlaneField:
    # Y9飞机
    def __init__(self, args):
        # self.args = args
        self.plane_nums = args.plane_nums
        self.planes = self.gen_planes()

    def gen_planes(self):
        planes = []
        for id in range(self.plane_nums):
            planes.append(PLANE(id))
        return planes

class RedSub(Unit):
    def __init__(self,id):
        self.id = id
        self.lat = None
        self.lon = None
        self.last_lat = None
        self.last_lon = None
        self.height = 0
        self.vel = 0
        self.mileage = 0
        self.course = 0
        self.task_area = Position()
        self.type = None
    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon
        self.last_height = self.height

class Red_Sub:
    def __init__(self, args):
        self.red_sub_nums = args.red_sub_nums
        self.red_subs = self.gen_red_subs()

    def gen_red_subs(self):
        red_subs = []
        for id in range(self.red_sub_nums):
            red_subs.append(RedSub(id))
        return red_subs

class Ship:
    def __init__(self, lat=None, lon=None, velocity=None, course=None, type=None):
        self.lat = lat
        self.lon = lon
        self.height = 0
        self.vel = velocity
        self.course = course
        self.type = type #型号名字
        self.mileage = 0  # 行驶里程

    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon
        self.last_height = self.height

class Plane:
    def __init__(self, lat=None, lon=None, height=None, velocity=None, course=None, type=None):
        self.lat = lat
        self.lon = lon
        self.height = height
        self.vel = velocity
        self.course = course
        self.type = type#型号名字
    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon
        self.last_height = self.height


class RedGlobalObservation():
    def __init__(self, args):
        # self.args = args
        self.simtime = 0
        self.uav_field = UAVField(args)
        self.usv_field = USVField(args)
        self.report = REPORT(args)





        self.plane_field = PlaneField(args)
        self.red_sub_field = Red_Sub(args)
        self.frigate_field = [Ship(args) for _ in range(args.red_frigate_nums)]
        self.maritime_ship_field = [Ship(args) for _ in range(args.red_maritime_ship_nums)]
        self.J20_field = [Plane(args) for _ in range(args.red_J20_nums)]
        self.H6_field = [Plane(args) for _ in range(args.red_H6_nums)]
        self.Elec_plane_field = [Plane(args) for _ in range(args.red_Elec_plane_nums)]




        self.entry_point = Position()
        self.call_point = [Position() for _ in range(args.submarine_nums)]
        self.multi_sensor_img = None
        self.task_message = ""
        self.buoy_message = ""
        self.virtual = Position()
        self.last_report = Position()
        # self.report_success_time = [[0, 0] for _ in range(args.submarine_nums)] #上报成功次数
        # self.start_track_time = [[0, []] for _ in range(args.submarine_nums)]
        # self.report_time_minute = [[]for _ in range(args.submarine_nums)] #每分钟上报次数


        self.report_success_time = [0 for _ in range(args.submarine_nums)]  # 上报成功次数
        self.start_track_time = [None for _ in range(args.submarine_nums)]
        self.report_time_minute = [[] for _ in range(args.submarine_nums)]  # 每分钟上报次数

        self.vir_history = []
        self.vir_history_index = -1
        self.last_vir = [[], []]
        self.vir_his_pos = []  # 记录历史可疑位置
        self.report_plan = False
        self.sub_pos = []  # 持续追踪过程中记录潜艇的位置
        self.submarine_nums = args.submarine_nums
        self.report_open_time = 0
        self.combination_touch_time = 0
        self.virtual_touch_time = 0


    def clear_report_info(self):
        self.report_success_time = [0 for _ in range(self.submarine_nums)]  # 上报成功次数
        self.start_track_time = [None for _ in range(self.submarine_nums)]
        self.report_time_minute = [[] for _ in range(self.submarine_nums)]  # 每分钟上报次数

