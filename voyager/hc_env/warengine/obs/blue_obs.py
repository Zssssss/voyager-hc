import numpy as np
from ..obs.red_obs import Battery
from ..sensor import blue_sonar as Sonar
from ..sensor import sonar_class as passive_sonar

from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84

def thermocline_detect(sonar_height, thermocline, target_height):
    if (abs(sonar_height) < abs(thermocline) and abs(target_height) < abs(thermocline)) or (abs(sonar_height) >= abs(thermocline) and abs(target_height) >= abs(thermocline)):
        return True
    else:
        return False

def sonar_env(red_obs, blue_obs, env_obs, name="舰壳被动声呐", thermocline=-90, id=0):
    """生成声呐探测需要的声源信息"""
    target = []  # 需求生成多个声源，如主动声呐、渔船
    jammer_range = 4000
    for uav_id in range(red_obs.uav_field.uav_nums):
        for buoy in red_obs.uav_field.uavs[uav_id].buoys:
            if buoy.btype == 1:
                append = True
                if name == "舰壳被动声呐":
                    if (not thermocline_detect(blue_obs.submarines[id].height, thermocline, buoy.height)) or geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, buoy.lat, buoy.lon)['s12'] > 8_000: #舰壳被动声呐和声呐浮标在跃变层的一侧
                        append = False
                if name == "拖曳声呐":
                    if geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, buoy.lat, buoy.lon)['s12'] > blue_obs.blue_drag_sonar:
                        append = False
                if append:
                    target.append(Sonar.Target(sub_lat=buoy.lat, sub_lon=buoy.lon,
                                               sub_alt=buoy.height, v=0, f=500 + np.random.uniform(40, 80),
                                               P=10, target_density=10,
                                               sub_name="主动声呐"))

    for sub_id in range(red_obs.red_sub_field.red_sub_nums):
        red_sub = red_obs.red_sub_field.red_subs[sub_id]
        append = True
        if name == "舰壳被动声呐":
            if (not thermocline_detect(blue_obs.submarines[id].height, thermocline, red_sub.height)) or geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, red_sub.lat, red_sub.lon)['s12'] > 8_000: #舰壳被动声呐和声呐浮标在跃变层的一侧
                append = False
            if name == "拖曳声呐":
                if geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, red_sub.lat, red_sub.lon)['s12'] > blue_obs.blue_drag_sonar:
                    append = False
        if append:
            target.append(Sonar.Target(sub_lat=red_sub.lat, sub_lon=red_sub.lon,
                                       sub_alt=red_sub.height, v=red_sub.vel, f=1000 + np.random.uniform(40, 80),
                                       P=50, target_density=10,
                                       sub_name="潜艇"))

    for usv_id in range(red_obs.usv_field.usv_nums):
        usv = red_obs.usv_field.usvs[usv_id]
        append = True
        if name == "舰壳被动声呐":
            if (not thermocline_detect(blue_obs.submarines[id].height, thermocline, 0)) or geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, usv.lat, usv.lon)['s12'] > blue_obs.blue_sonar : #舰壳被动声呐和声呐浮标在跃变层的一侧
                append = False
            if name == "拖曳声呐":
                if geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, usv.lat, usv.lon)['s12'] > blue_obs.blue_drag_sonar:
                    append = False
                if blue_obs.submarines[id].height < thermocline:
                    append = False
        if append:
            target.append(Sonar.Target(sub_lat=usv.lat, sub_lon=usv.lon,
                                       sub_alt=0, v=usv.vel, f=400 + np.random.uniform(40, 80),
                                       P=40, target_density=10,
                                       sub_name="无人艇"))


    for i in range(env_obs.fishing_boat_nums):  # 渔船信息
        append = True
        if name == "舰壳被动声呐":
            if (not thermocline_detect(blue_obs.submarines[id].height, thermocline, 0)) or geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, env_obs.fishing_boats[i].lat, env_obs.fishing_boats[i].lon)['s12'] > blue_obs.blue_sonar :
                append = False
            if name == "拖曳声呐":
                if geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, env_obs.fishing_boats[i].lat, env_obs.fishing_boats[i].lon)['s12'] > blue_obs.blue_drag_sonar:
                    append = False
                if blue_obs.submarines[id].height < thermocline:
                    append = False
        if append:
            target.append(Sonar.Target(sub_lat=env_obs.fishing_boats[i].lat, sub_lon=env_obs.fishing_boats[i].lon,
                                       sub_alt=0, v=-10, f=200 + np.random.uniform(10, 20), P=1, target_density=5,
                                       sub_name="渔船"))


    for i in range(env_obs.cargo_ship_nums):  # 货轮信息
        append = True
        if name == "舰壳被动声呐":
            if (not thermocline_detect(blue_obs.submarines[id].height, thermocline, 0)) or geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, env_obs.cargo_ships[i].lat, env_obs.cargo_ships[i].lon)['s12'] > blue_obs.blue_sonar :
                append = False
            if name == "拖曳声呐":
                if geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, env_obs.cargo_ships[i].lat, env_obs.cargo_ships[i].lon)['s12'] > blue_obs.blue_drag_sonar:
                    append = False
                if blue_obs.submarines[id].height < thermocline:
                    append = False
        if append:
            target.append(Sonar.Target(sub_lat=env_obs.cargo_ships[i].lat, sub_lon=env_obs.cargo_ships[i].lon,
                                       sub_alt=0, v=-17, f=300 + np.random.uniform(20, 30), P=5, target_density=10,
                                       sub_name="货轮"))

    for i in range(env_obs.fish_nums):  # 鱼群信息
        append = True
        if name == "舰壳被动声呐":
            if (not thermocline_detect(blue_obs.submarines[id].height, thermocline, 0)) or geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, env_obs.fishs[i].lat, env_obs.fishs[i].lon)['s12'] > blue_obs.blue_sonar:
                append = False
            if name == "拖曳声呐":
                if geod.Inverse(blue_obs.submarines[id].lat, blue_obs.submarines[id].lon, env_obs.fishs[i].lat, env_obs.fishs[i].lon)['s12'] > blue_obs.blue_drag_sonar:
                    append = False
        if append:
            target.append(Sonar.Target(sub_lat=env_obs.fishs[i].lat, sub_lon=env_obs.fishs[i].lon,
                                       sub_alt=env_obs.fishs[i].height, v=-7, f=100 + np.random.uniform(8, 15), P=3,
                                       target_density=10,
                                       sub_name="鱼群"))

    env = Sonar.Environment(target)
    return env


class Blue_sonar:
    """舰壳被动声呐"""
    def __init__(self):
        self.statu = 0
        self.touch = False
        self.target_course = None #被动声呐：目标的方位，比如两个目标[xx, xx]
        self.target_data = None#声呐的接收信号{'x': array(), 'y': array()}
        self.target_feature = None#被动声呐：加入存在两个目标，[{'theta': xx, 'f': xx, 'p_recevied': xx}, {'theta': xx, 'f': xx, 'p_recevied': xx}]。
        self.self_pos = {"lat": None, "lon": None, "height": None}
        self.open_time = 0
        self.touch_time = 0

    def result_clear(self):
        self.touch = False
        self.target_feature = None
        self.target_data = None
        self.target_course = None
        self.self_pos = {"lat": None, "lon": None, "height": None}


    def sensor_detect(self, red_obs, blue_obs, env_obs, thermocline_height=-90, id=0, sensor_data=False):
        """thermocline:跃变层距离"""
        env = sonar_env(red_obs, blue_obs, env_obs, name="舰壳被动声呐", thermocline=thermocline_height, id=id)
        sonar = passive_sonar.passive_sonar(sonar_lat=blue_obs.submarines[id].lat, sonar_lon=blue_obs.submarines[id].lon, sonar_alt=blue_obs.submarines[id].height, v=blue_obs.submarines[id].vel, env=env,
                                    name="被动声呐")
        res = sonar.passive_detect(plt_show=sensor_data)
        if not res:
            self.touch = False
            self.target_feature = None
            self.target_data = None
            self.target_pos = None
            self.course = None
            self.self_pos = {"lat": blue_obs.submarines[id].lat, "lon": blue_obs.submarines[id].lon, "height": blue_obs.submarines[id].height}
        else:
            self.touch = True
            self.self_pos = {"lat": blue_obs.submarines[id].lat, "lon": blue_obs.submarines[id].lon, "height": blue_obs.submarines[id].height}
            if sensor_data:
                self.target_data = {"x": res[1], "y": res[2]}
                self.target_feature = res[0]
            else:
                self.target_data = None
                self.target_feature = res
            for i in range(len(self.target_feature)):
                self.target_feature[i]['theta'] = passive_sonar.theta_to_angle(
                    self.target_feature[i]['theta'])  # 方位角变化为geod适用形式
            self.target_pos = None
            self.target_course = [info['theta'] for info in self.target_feature]


class Drag_Sonar:
    def __init__(self, hydrophone_num=3, hydrophone_dis=2):
        """被动拖曳声呐--探测范围100km"""
        self.statu = 0  # 0:未激活，1:激活
        self.touch = False  # 是否探测到可疑目标
        # {"theta": 信号方位（measured in radians）, "f": 信号频率（hz),  "p_recevied": 接收信号声强级}
        self.target_feature = None  # 每个水听器对每个目标的特征信息， 比如两个目标时结果{'sonar0': [{'theta': xx, 'f': xx, 'p_recevied': xx}, {'theta': xx, 'f': xx, 'p_recevied': xx}], 'sonar1': ..., 'sonar2': ...}，theta表示目标到当前水听器的角度，已经转化成-180到180的形式
        self.target_data = None# 每个水听器接收到的信号{'sonar0': {'x': array(), 'y': array()}, 'sonar1': .., 'sonar2': ..}
        self.target_pos = None#多个目标的位置信息，比如两个目标时结果：[{'lat': xx, 'lon': xx}, {'lat': xx, 'lon': xx}],一个目标时结果：[{'lat': xx, 'lon': xx}
        self.hydrophone_num = hydrophone_num  # 拖曳声呐探测水听器阵元个数
        self.hydrophone_dis = hydrophone_dis  # 阵元间隔
        self.theta_rope = None  # 绳缆与海平面夹角
        self.rope_len = 800  # 绳缆长度
        self.theta_hydrophone = None  # 拖曳阵与绳缆夹角
        self.hydrophone_pos = None  # 记录水听器的经纬度 比如有三个水听器组成的拖曳浮标，结果：'sonar0': {'lat': xx, 'lon': xx, 'height': xx}, 'sonar1': {'lat': xx, 'lon': xx, 'height': xx}, 'sonar2': {'lat': xx, 'lon': xx, 'height': xx}}
        self.self_pos = {"lat": None, "lon": None, "height": None}
        self.open_time = 0 #发送message用，防止重复发送
        self.touch_time = 0

    def result_clear(self):
        self.touch = False
        self.target_feature = None
        self.target_data = None
        self.target_pos = None
        self.hydrophone_pos = None
        self.self_pos = {"lat": None, "lon": None, "height": None}

    def sonar_control(self, statu, theta_rope=10, rope_len=800, theta_hydrophone=20):
        self.statu = statu
        self.theta_rope = theta_rope
        self.rope_len = rope_len
        self.theta_hydrophone = theta_hydrophone

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
        if self.statu:
            env = sonar_env(red_obs, blue_obs, env_obs, name="拖曳声呐", id=id)
            Drag_array_sonar = Sonar.Passive_drag_array_sonar(env, rope_len=self.rope_len,
                                                              hydrophone_num=self.hydrophone_num,
                                                              hydrophone_dis=self.hydrophone_dis)  # rope_len:绳缆长度、hydrophone_num：阵元个数、hydrophone_dis：阵元间隔
            res_Drag_array = Drag_array_sonar.analyse_location(JT_lat=blue_obs.submarines[id].lat,
                                                               JT_lon=blue_obs.submarines[id].lon, JT_alt=blue_obs.submarines[id].height, v_lon=10,
                                                               v_lat=-30, theta_rope=self.theta_rope,
                                                               theta_hydrophone=self.theta_hydrophone,
                                                               thermocline_height=thermocline_height, plt_show=sensor_data)
            if not res_Drag_array:
                self.touch = False
                self.target_feature = None
                self.target_data = None
                self.target_pos = None
                self.hydrophone_pos = None
                self.self_pos = {"lat": blue_obs.submarines[id].lat, "lon": blue_obs.submarines[id].lon, "height": blue_obs.submarines[id].height}
            else:
                self.touch = True
                self.self_pos = {"lat": blue_obs.submarines[id].lat, "lon": blue_obs.submarines[id].lon,
                                 "height": blue_obs.submarines[id].height}
                if sensor_data:
                    self.target_data = res_Drag_array['sonar_info']['data']
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



class Jammer:
    def __init__(self, lat, lon, height, start_time):
        self.lat = lat
        self.lon = lon
        self.height = height
        self.lure_flag = False  # 是否干扰到对手
        self.start_time = start_time
        self.alive_time = 35 * 60  # 剩余生活时间
        self.dead = False if self.alive_time > 0 else True
        self.course = 0
        self.wave = None

    def update(self, now_time):
        self.alive_time = max(35 * 60 - (now_time - self.start_time), 0)  # 剩余生活时间
        self.dead = False if self.alive_time > 0 else True


# 潜艇声诱饵
class acoustic_bait:
    def __init__(self, lat, lon, height, velocity, course, start_time):
        self.lat = lat
        self.lon = lon
        self.last_lat = lat
        self.last_lon = lon
        self.height = height
        self.vel = velocity
        self.course = course
        self.attract = False  # 是否引诱到对手
        self.start_time = start_time
        self.alive_time = 40 * 60  # 剩余工作时间
        self.dead = False if self.alive_time > 0 else True

    def update(self, now_time):
        self.alive_time = max(40 * 60 - (now_time - self.start_time), 0)  # 剩余生活时间
        self.dead = False if self.alive_time > 0 else True
        if not self.dead:
            # 更新速度和方向
            g = geod.Direct(lat1=self.lat , lon1=self.lon, azi1=self.course, s12=self.vel*0.5144)
            self.lat = g["lat2"]
            self.lon = g["lon2"]



# 潜艇潜望镜
class Periscope:
    def __init__(self, periscope_range):
        self.detect_range = None  # 单位m
        self.long = 12  # 镜筒长度，单位m
        self.statu = 0
        self.result = []
        self.touch = False
        self.detect_range = periscope_range
        self.open_time = 0
        self.touch_time = 0

    def result_clear(self):
        self.touch = False
        self.result = []

    def sensor_detect(self, red_obs, env_obs, lat, lon, height):
        self.result = []
        self.touch = False
        Periscope_height = height + self.long
        if Periscope_height >= 0: #潜望镜可以探测到海平面外面
            for id in range(red_obs.uav_field.uav_nums):
                uav = red_obs.uav_field.uavs[id]
                g = geod.Inverse(lat, lon, uav.lat, uav.lon)
                dis = np.sqrt(g['s12'] ** 2 + (Periscope_height - uav.height) ** 2)
                if dis <= self.detect_range:
                    self.result.append({"uav":{"lat":uav.lat + np.random.uniform(-0.01, 0.01),"lon":uav.lon+ np.random.uniform(-0.01, 0.01),"height":uav.height + np.random.uniform(-20, 20)}})


            for id in range(red_obs.usv_field.usv_nums):
                usv = red_obs.usv_field.usvs[id]
                g = geod.Inverse(lat, lon, usv.lat, usv.lon)
                dis = g['s12']
                if dis <= self.detect_range:
                    self.result.append({"usv":{"lat":usv.lat + np.random.uniform(-0.01, 0.01),"lon":usv.lon+ np.random.uniform(-0.01, 0.01),"height":0 + np.random.uniform(-5, 5)}})

            for i in range(env_obs.fishing_boat_nums):  # 渔船信息
                fishing_boat = env_obs.fishing_boats[i]
                g = geod.Inverse(lat, lon, fishing_boat.lat,fishing_boat.lon)
                dis = g['s12']
                if dis <= self.detect_range:
                    self.result.append({"fishing_boat": {"lat": fishing_boat.lat + np.random.uniform(-0.01, 0.01),
                                                "lon": fishing_boat.lon + np.random.uniform(-0.01, 0.01),
                                                "height": 0 + np.random.uniform(-5, 5)}})


            for i in range(env_obs.cargo_ship_nums):  # 货轮信息
                cargo_ship = env_obs.cargo_ships[i]
                g = geod.Inverse(lat, lon, cargo_ship.lat, cargo_ship.lon)
                dis = g['s12']
                if dis <= self.detect_range:
                    self.result.append({"cargo_ship": {"lat": cargo_ship.lat + np.random.uniform(-0.01, 0.01),
                                                         "lon":cargo_ship.lon + np.random.uniform(-0.01, 0.01),
                                                         "height": 0 + np.random.uniform(-5, 5)}})
        if len(self.result) != 0:
            self.touch = True



class Position:
    def __init__(self, lat=None, lon=None, alt=None):
        self.lat = alt
        self.lon = alt
        self.alt = alt
        self.target_type = None

class Ship:
    def __init__(self, lat=None, lon=None, velocity=None, course=None, type=None):
        self.lat = lat
        self.lon = lon
        self.height = 0
        self.vel = velocity
        self.course = course
        self.type = type#型号名字
        self.mileage = 0
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
        self.mileage = 0
    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon
        self.last_height = self.height

"""
蓝方全局态势
"""


class BlueGlobalObservation:
    def __init__(self, args):
        self.simtime = 0
        # self.args = args
        self.blue_sonar = args.blue_sonar #蓝方舰壳声呐最大探测范围
        self.thermocline = args.thermocline #跃变层位置
        self.submarine_nums = args.submarine_nums
        self.blue_drag_sonar = args.blue_drag_sonar#蓝方拖曳声呐最大探测范围
        self.patrol_ship_nums = args.blue_patrol_ship_nums
        self.destroyer_nums = args.blue_destroyer_nums
        self.P1_plane_nums = args.blue_P1_plane_nums
        self.F15_plane_nums = args.blue_F15_plane_nums
        self.Elec_plane_nums = args.blue_Elec_plane_nums
        # self.usv_nums = self.args.usv_nums

        # self.usvs = [USV() for _ in range(self.usv_nums)]
        self.submarines = [Submarine(args) for _ in range(self.submarine_nums)]
        self.patrol_ships = [Ship() for _ in range(self.patrol_ship_nums)]
        self.destroyers = [Ship() for _ in range(self.destroyer_nums)]
        self.task_point = [Position() for _ in range(self.submarine_nums)]
        self.P1_plane = [Plane() for _ in range(self.P1_plane_nums)]
        self.F15_plane = [Plane() for _ in range(self.F15_plane_nums)]
        self.Elec_plane = [Plane() for _ in range(self.Elec_plane_nums)]



class Unit:
    def __init__(self):
        self.lat = 0
        self.lon = 0
        self.height = 0
        self.course = 0
        self.vel = 0

"""
潜艇态势类 （实体类） 可以探测到主动声呐浮标
"""


class Submarine(Unit):
    def __init__(self, args):
        super(Submarine, self).__init__()
        # self.args = args
        self.drag_sonar = Drag_Sonar() #拖曳声呐
        self.sonar = Blue_sonar()
        self.type = None #潜艇名字型号
        # self.active_sonar = Sonar(agent_type="submarine", sonar_type=1)
        self.jammer_nums = 2  # 干扰器数量
        self.jammers = []
        self.bait_nums = 2  # 声诱饵数量
        self.bait = []

        self.snorkel = False  # 通气管状态，为了充电
        self.snorkel_open_time = 0
        self.periscope = Periscope(args.periscope_range)

        # 潜艇续航信息
        if args.Sub_params_Init_Battery == '1':  # 电量充足
            battery_ratio = np.random.uniform(75, 90)
        else:  # 电量不充足
            battery_ratio = np.random.uniform(50, 75)
        self.battery = Battery(en_time_max=4.5 * 24 * 60, battery_max=70, battery_ratio=battery_ratio, en_dis_max=500)

        self.task_message = ""
        self.lat = None
        self.lon = None
        self.course = None
        self.height = None
        self.phi = 0
        self.theta = 0
        self.psi = 0
        self.vel = 0
        self.mileage = 0  # 潜艇行驶里程
        self.sub_field = {"min_lat": 0, "max_lat": 0, "min_lon": 0, "max_lon": 0}

    def update_params(self):
        self.last_lat = self.lat
        self.last_lon = self.lon
        self.last_height = self.height
        self.last_phi = self.phi  # 滚转角
        self.last_theta = self.theta  # 俯仰角
        self.last_psi = self.psi  # 偏航角
        self.last_vel = self.vel
        self.last_course = self.course


