from asyncio import sleep
from typing import List

from geographiclib.geodesic import Geodesic

from ..commands.plane_command import PlaneCommand, RedObjType
from ..obs.env_obs import EnvGlobalObservation
from ..obs.red_obs import RedGlobalObservation

import numpy as np

geod = Geodesic.WGS84
from pyproj import Transformer


# 点的类型
class PositionType:
    MAG = 0
    PHOTO = 1
    RADAR = 2
    VIRTUAL = 3
    TOUCH = 4


def lla_to_xyz(lat, lon, alt):
    transprojr = Transformer.from_crs(
        "EPSG:4326",
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True)
    x, y, z = transprojr.transform(lon, lat, alt, radians=False)
    return x, y, z


def xyz_to_lla(x, y, z):
    transproj = Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, alt = transproj.transform(x, y, z, radians=False)
    return lon, lat, alt


# 点的类型
class Position:
    def __init__(self, lat=None, lon=None, type=None, simtime=None, locat_buoys=None, course=None, btype=None,
                 target_type=None):
        self.lat = lat
        self.lon = lon
        self.type = type
        self.simtime = simtime
        self.locat_buoys = locat_buoys
        self.course = course
        self.btype = btype
        self.target_type = target_type


class UAVAssist:
    def __init__(self):
        self.id = 0
        # self.buoy_target = []
        self.buoys_index = []
        self.mode = 0
        self.buoys_target = []  # 目标浮标列表
        self.buoys_index = 0  # 目标浮标id
        self.vir_point = None  # 多枚声呐浮标精确定位
        self.attack_idx = 90  # 攻击浮标id
        self.touch_buoy = None  # touch的浮标对象
        self.active_idx = 50
        # self.vir_history = []
        self.patrol_index = [[], []]  # 八字形巡逻路径点
        self.patrol_plan = True  # 是否需要规划八字巡逻路径
        self.patrol_start_pos = []
        self.patrol_start_index = 0
        self.patrol_start = False  # 是否到达八字起始点
        self.patroll_start_pos_plan = True  # 是否规划八字起始点位置
        self.buoys_help = [False, []]  # 是否需要别的无人机帮忙在某位置投放声呐浮标
        self.buoy_array_pos_plan = True  # 是否规划浮标阵列起始点位置
        self.com_buoys_index = 0  # 检查投放浮标是否死亡id
        self.com_buoys = False  # 上一步是否投放了浮标
        self.center_arrive = False
        self.obs_type = None
        self.obs_type_times = 0
        self.centor_lat_focus = None
        self.centor_lon_focus = None
        self.center_plan = True
        self.obs_turn_points = []
        self.obs_array_finish = True
        self.obs_turn = False
        self.vit_times = 0
        self.target_arrive = False
        self.centor_pos = Position()
        self.farthest_pos = [None, None]
        self.obs_angle = None


# 状态
class Mode:
    GO_TO_CALL_POINT = 0  # 前往应召地点
    PLACE_BUOYS = 1  # 投放浮标
    GO_TO_USV_REILL = 2  # 补充浮标
    PATROL = 3  # 巡逻
    TRACK = 4  # 跟踪 发现有touch
    IDENT = 5
    REPORT = 6


class PlaneDecision:
    def __init__(self):
        # self.mode = 0 # 0:飞向应找点 1：规划浮标阵型并 投放声呐浮标 2:有touch开始追击 3：有virtual开始追击

        # 当前模式
        self.uav_assists = []
        self.usv_modes = []

        self.uav_message = []
        self.usv_message = []
        self.message_sensor_infos = ""
        self.log_flag = True  # 日志标志位

        self.call_point = None

        # USV相关参数设置
        self.screw_r = 10000  # 螺旋巡逻半径
        self.screw_angle = 0  # 螺旋角度


    # 打印日志
    def log(self, msg):
        if self.log_flag:
            print(msg)

    def patrol_path(self, start_lat, start_lon, centor_lat, centor_lon, width, theta1=0.5 * np.pi, theta2=2.5 * np.pi,
                    num=100):
        """"无人机从start位置到farthest位置的八字路径
        其中start为起点, farthest为中心点，即八字的交点
        width为八字的宽度"""
        g = geod.Inverse(start_lat, start_lon, centor_lat, centor_lon)
        t = np.linspace(theta1, theta2, num)
        if width is None:
            width = np.clip(1e7 / g['s12'], 800, 3000)
        x = g['s12'] * np.sin(t) / (1 + np.cos(t) ** 2)
        y = width * np.sin(t) * np.cos(t) / (1 + np.cos(t) ** 2)
        if g['azi1'] >= 0:
            theta = np.deg2rad(270 - g['azi1'])
        else:
            theta = np.deg2rad(-g['azi1'] - 90)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x_rotated, y_rotated = np.dot(rotation_matrix, np.vstack((x, y)))
        # 转换到经纬度
        psi = np.arctan2(x_rotated, y_rotated) * 180 / np.pi  # 角度
        for i in range(np.size(x_rotated)):
            g = geod.Direct(lat1=centor_lat, lon1=centor_lon,
                            s12=np.linalg.norm(np.array([x_rotated[i], y_rotated[i]])),
                            azi1=psi[i])
            x_rotated[i], y_rotated[i] = g['lat2'], g['lon2']

        return x_rotated, y_rotated

    # 计算两个方位的夹角 以及a1 相对于a0是顺时针还是逆时针 1 是顺时针 0 是逆时针
    def cal_subangle(self, a0, a1):
        a0, a1 = a0 % 360, a1 % 360
        delta_angle = abs(a0 - a1)
        dir = 1
        if delta_angle <= 180:
            if a1 >= a0:
                dir = 1
            else:
                dir = -1
        else:
            delta_angle = 360 - delta_angle
            if 0 <= a1 <= 180:
                dir = 1
            else:
                dir = -1
        return delta_angle, dir

    def cal_target(self, buoy1, buoy2):
        """两个不同型号的定向浮标，计算目标位置"""
        if buoy1.btype == '63':
            s1_max = 1200
        else:
            s1_max = 3000

        if buoy2.btype == '63':
            s2_max = 1200
        else:
            s2_max = 3000
        if abs(buoy1.course) == abs(buoy2.course):
            return None, None

        for s in range(s1_max + 500, 50):
            g = geod.Direct(buoy1.lat, buoy1.lon, s12=s, azi1=buoy1.course)
            lat, lon = g['lat2'], g['lon2']
            g = geod.Inverse(buoy2.lat, buoy2.lon, lat, lon)
            if g['azi1'] <= 0.3:
                return lat, lon

        return None, None

    def cal_target_new(self, buoy, callpoint, s12):
        if buoy.btype == '63':
            s1_max = 1200
        else:
            s1_max = 3000

        g = geod.Direct(buoy.lat, buoy.lon, s12=s1_max, azi1=buoy.course)
        lat, lon = g['lat2'], g['lon2']
        g = geod.Inverse(callpoint.lat, callpoint.lon, lat, lon)
        course = g['azi1']
        g = geod.Direct(lat, lon, s12=s12, azi1=course)
        return g['lat2'], g['lon2']

    def patrol_decision(self, id, uav, centor_pos=None, type=0, width=6_000, num=18):
        # 八字形巡逻
        """"
        id:无人机编号
        centor_pos:八字巡逻的中心的
        type=0: 圆形八字巡逻
        type=1: 直线八字巡逻 todo
        width：八字形的宽度
        num：圆形搜索八字形的个数"""
        command = None
        # 确定八字形起始点
        # if p_call_g["s12"] <= 20_000 and self.uav_assists[id].mode != Mode.PATROL:
        #     self.uav_assists[id].mode = Mode.PATROL  # 巡逻

        if self.uav_assists[id].patroll_start_pos_plan and self.uav_assists[id].mode == Mode.PATROL and type == 0:
            # 确定八字形起始点,采用圆形，顺时针旋转
            p_call_g_inv = geod.Inverse(centor_pos.lat, centor_pos.lon, uav.lat, uav.lon)
            for angle in np.arange(p_call_g_inv['azi1'], 360 + p_call_g_inv['azi1'], 360 / num):
                g_ = geod.Direct(lat1=centor_pos.lat, lon1=centor_pos.lon,
                                 s12=p_call_g_inv["s12"], azi1=angle)
                self.uav_assists[id].patrol_start_pos.append([g_['lat2'], g_['lon2']])
                self.uav_assists[id].patroll_start_pos_plan = False  # 起始点规划完毕

        if self.uav_assists[id].mode == Mode.PATROL and not self.uav_assists[
            id].patroll_start_pos_plan:  # 无人机位于巡逻状态并且已经有规划好的八字路径
            g = geod.Inverse(uav.lat, uav.lon, self.uav_assists[id].patrol_start_pos[
                self.uav_assists[id].patrol_start_index % len(self.uav_assists[id].patrol_start_pos)][0],
                             self.uav_assists[id].patrol_start_pos[
                                 self.uav_assists[id].patrol_start_index % len(
                                     self.uav_assists[id].patrol_start_pos)][1])

            if g["s12"] <= 400 or self.uav_assists[id].patrol_start:  # 到达了起始点或者开始画八字形
                self.uav_assists[id].patrol_start = True  # 到达了一个八字形起始点
                if self.uav_assists[id].patrol_plan:  # 规划一个八字的路径
                    # print('uav id{}，开始规划第{}个八字形'.format(id, self.uav_assists[id].patrol_start_index))
                    # 规划八字路径
                    x, y = self.patrol_path(self.uav_assists[id].patrol_start_pos[
                                                self.uav_assists[id].patrol_start_index % len(
                                                    self.uav_assists[id].patrol_start_pos)][0],
                                            self.uav_assists[id].patrol_start_pos[
                                                self.uav_assists[id].patrol_start_index % len(
                                                    self.uav_assists[id].patrol_start_pos)][1], centor_pos.lat,
                                            centor_pos.lon,
                                            width)
                    self.uav_assists[id].patrol_index = [x, y]
                    self.uav_assists[id].patrol_plan = False

                if not self.uav_assists[id].patrol_plan:
                    # 执行八字路径
                    g = geod.Inverse(uav.lat, uav.lon, self.uav_assists[id].patrol_index[0][0],
                                     self.uav_assists[id].patrol_index[1][0])
                    if g["s12"] <= 400:
                        # 已经走到八字中的一个点
                        self.uav_assists[id].patrol_index[0] = self.uav_assists[id].patrol_index[0][
                                                               1:]  # 在一个八字中，去掉已经完成的点
                        self.uav_assists[id].patrol_index[1] = self.uav_assists[id].patrol_index[1][1:]
                        if len(self.uav_assists[id].patrol_index[0]) != 0:
                            g = geod.Inverse(uav.lat, uav.lon, self.uav_assists[id].patrol_index[0][0],
                                             self.uav_assists[id].patrol_index[1][0])  ## 在一个八字中，完成一个点后去下一个点
                        else:
                            # print('uav id{}，执行完第{}个八字形'.format(id, self.uav_assists[id].patrol_start_index))
                            self.uav_assists[id].patrol_plan = True
                            self.uav_assists[id].patrol_start_index += 1  # 到规划好的下一个八字开始点
                            self.uav_assists[id].patrol_start = False  # 没有到达八字形起始点

            command = PlaneCommand.move_control(obj_type=RedObjType.UAV, id=id,
                                                velocity=500, height=500, course=g["azi1"])
        return command

    def buoy_array_decision(self, id, red_obs, buoy_num=30):
        uav = red_obs.uav_field.uavs[id]
        call_point = red_obs.call_point
        simtime = red_obs.simtime
        command = None
        if self.uav_assists[id].buoy_array_pos_plan:
            p_call_g_inv = geod.Inverse(call_point.lat, call_point.lon, uav.lat, uav.lon)
            for angle in np.arange(p_call_g_inv['azi1'], 360 + p_call_g_inv['azi1'], 360 // buoy_num):
                target_buoy_g = geod.Direct(lat1=call_point.lat, lon1=call_point.lon,
                                            s12=15000 * (id * 0.2 + 1), azi1=angle)
                self.uav_assists[id].buoys_target.append(target_buoy_g)
                self.uav_assists[id].buoy_array_pos_plan = False  # 浮标阵列规划完毕

        if self.uav_assists[id].mode == Mode.PLACE_BUOYS and not self.uav_assists[id].buoy_array_pos_plan:
            target_buoy = self.uav_assists[id].buoys_target[
                self.uav_assists[id].buoys_index % len(self.uav_assists[id].buoys_target)]
            g = geod.Inverse(uav.lat, uav.lon, target_buoy["lat2"], target_buoy["lon2"])
            # self.log('Time: {} uav call dis: {}'.format(simtime, g['s12']))
            if self.uav_assists[id].buoys_index < len(self.uav_assists[id].buoys_target):
                if g["s12"] <= 400:
                    if red_obs.uav_field.uavs[id].buoy_62_nums > 0:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=62,
                                                         channel=self.uav_assists[id].buoys_index, height=-175,
                                                         total=len(self.uav_assists[id].buoys_target))
                    elif red_obs.uav_field.uavs[id].buoy_63_nums > 0:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=63,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175,
                                                         total=len(self.uav_assists[id].buoys_target))
                    elif red_obs.uav_field.uavs[id].buoy_67_nums > 0:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=67,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175,
                                                         total=len(self.uav_assists[
                                                                       id].buoys_target))  # 放置被动声呐浮标阵列
                    self.uav_assists[id].buoys_index += 1
                    self.uav_message.append("放置第{}个被动声呐浮标".format(self.uav_assists[id].buoys_index))
                else:
                    command = PlaneCommand.move_control(obj_type=RedObjType.UAV, id=id,
                                                        velocity=500, height=500, course=g["azi1"])
            else:
                # 目标声呐已经全部投完，如果有浮标死亡，进行补投,如果自己没有浮标了找别的无人机帮忙补投，自己进行巡逻
                if self.uav_assists[id].com_buoys_index < len(red_obs.uav_field.uavs[id].buoys):
                    buoy = red_obs.uav_field.uavs[id].buoys[self.uav_assists[id].com_buoys_index]
                    if buoy.dead:
                        # self.log('Time: {} 死亡的声呐浮标channel: {}'.format(simtime, buoy.channel))
                        # print('自己还有的被动声呐数量', red_obs.uav_field.uavs[id].buoy_62_nums,
                        #       red_obs.uav_field.uavs[id].buoy_63_nums,
                        #       red_obs.uav_field.uavs[id].buoy_67_nums)
                        if red_obs.uav_field.uavs[id].buoy_62_nums > 0 or red_obs.uav_field.uavs[id].buoy_63_nums > 0 or \
                                red_obs.uav_field.uavs[id].buoy_67_nums > 0:
                            # 还剩被动声呐
                            g = geod.Inverse(uav.lat, uav.lon, buoy.lat, buoy.lon)
                            # self.log('Time: {} uav call dis: {}'.format(simtime, g['s12']))
                            if g["s12"] <= 400:
                                if red_obs.uav_field.uavs[id].buoy_62_nums > 0:
                                    command = PlaneCommand.drop_buoy(id=id, buoy_type=62,
                                                                     channel=buoy.channel,
                                                                     height=-175,
                                                                     total=len(self.uav_assists[id].buoys_target))
                                elif red_obs.uav_field.uavs[id].buoy_63_nums > 0:
                                    command = PlaneCommand.drop_buoy(id=id, buoy_type=63,
                                                                     channel=buoy.channel,
                                                                     height=-175,
                                                                     total=len(self.uav_assists[id].buoys_target))
                                elif red_obs.uav_field.uavs[id].buoy_67_nums > 0:
                                    command = PlaneCommand.drop_buoy(id=id, buoy_type=67,
                                                                     channel=buoy.channel,
                                                                     height=-175,
                                                                     total=len(self.uav_assists[id].buoys_target))
                                    self.log('浮标channel: {}，补投67声呐浮标'.format(buoy.channel))
                                self.uav_assists[id].com_buoys_index += 1
                            else:
                                command = PlaneCommand.move_control(obj_type=RedObjType.UAV, id=id,
                                                                    velocity=500, height=500, course=g["azi1"])
                        else:
                            self.uav_assists[id].buoys_help[0] = True
                            self.uav_assists[id].buoys_help[1].append(buoy) if buoy not in \
                                                                               self.uav_assists[id].buoys_help[
                                                                                   1] else None
                            # print('得靠别人了，help: ', self.uav_assists[id].buoys_help)
                            self.uav_assists[id].com_buoys_index += 1
                    else:
                        self.uav_assists[id].com_buoys_index += 1
                else:
                    self.uav_assists[id].mode = Mode.PATROL
        return command

    def put_buoys(self, id, uav, buoy_all, put):
        min_dist = np.inf
        command = None
        min_68 = np.inf
        min_62 = np.inf
        min_63 = np.inf
        min_67 = np.inf
        for buoy in buoy_all:
            if not buoy.dead:
                buoy_g = geod.Inverse(uav.lat, uav.lon, buoy.lat, buoy.lon)
                if buoy_g['s12'] < min_dist:
                    min_dist = buoy_g['s12']  # 确保声呐浮标之间的距离不要太小

                if buoy.btype == "68":
                    if buoy_g['s12'] < min_68:
                        min_68 = buoy_g['s12']

                if buoy.btype == "62":
                    if buoy_g['s12'] < min_62:
                        min_62 = buoy_g['s12']

                if buoy.btype == "63":
                    if buoy_g['s12'] < min_63:
                        min_63 = buoy_g['s12']

                if buoy.btype == "67":
                    if buoy_g['s12'] < min_67:
                        min_67 = buoy_g['s12']

        if min_dist > 800:
            if put == 2:
                # 投放主动声纳浮标拦截
                if min_68 > 5000 and uav.buoy_68_nums > 0:
                    command = PlaneCommand.drop_buoy(id=id, buoy_type=68,
                                                     channel=self.uav_assists[id].buoys_index,
                                                     height=-175)
            elif put == 0:
                # 投放被动声纳浮标
                if uav.buoy_67_nums > 0:
                    if min_67 > 4000:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=67,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175)
                elif uav.buoy_63_nums > 0:
                    if min_63 > 2000:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=63,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175)
                elif uav.buoy_62_nums > 0:
                    if min_62 > 2000:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=62,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175)
            elif put == 1:
                if uav.buoy_67_nums > 0:
                    if min_67 > 1500:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=67,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175)
                elif uav.buoy_63_nums > 0:
                    if min_63 > 800:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=63,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175)
                elif uav.buoy_62_nums > 0:
                    if min_62 > 800:
                        command = PlaneCommand.drop_buoy(id=id, buoy_type=62,
                                                         channel=self.uav_assists[id].buoys_index,
                                                         height=-175)

        return command


    def intercept_array(self, id, red_obs, center_lat, center_lon, call_point, buoy_all, obs_type_start=0, put=2,
                        straight_len=10_000, num=20, angle=None,):
        """阻拦阵列，放主动声纳"""
        uav = red_obs.uav_field.uavs[id]
        obs_type = list(np.arange(0, 4))
        command = None
        if self.uav_assists[id].obs_type is None:
            self.uav_assists[id].obs_type = obs_type_start

        if angle is None:
            g = geod.Inverse(call_point.lat, call_point.lon, center_lat, center_lon)
            angle = g["azi1"]#预估潜艇的方向

        self.uav_assists[id].obs_type = self.uav_assists[id].obs_type % len(obs_type)

        if self.uav_assists[id].obs_type in [0, 2]:
            start_lat = center_lat
            start_lon = center_lon
            g = geod.Inverse(start_lat, start_lon, uav.lat, uav.lon)
            if self.uav_assists[id].obs_type == 0:
                angle_ = angle + 90
            else:
                angle_ = angle - 90
            if g['s12'] > straight_len:
                self.uav_assists[id].farthest_pos = [uav.lat, uav.lon]
                self.uav_assists[id].obs_type += 1
                self.uav_assists[id].obs_type_times += 1
            else:
                if put is not None and g['s12'] > 400:
                    command = self.put_buoys(id, uav, buoy_all, put=put)
                if command is None or put is None:
                    command = PlaneCommand.move_control(obj_type=RedObjType.UAV, id=id, velocity=650,
                                                        height=500,
                                                        course=angle_)

        if self.uav_assists[id].obs_type in [1, 3]:
            start_lat = self.uav_assists[id].farthest_pos[0]
            start_lon = self.uav_assists[id].farthest_pos[1]
            g = geod.Inverse(start_lat, start_lon, uav.lat, uav.lon)
            if self.uav_assists[id].obs_type == 3:
                angle_ = angle + 90
            else:
                angle_ = angle - 90
            if g['s12'] > straight_len:
                self.uav_assists[id].obs_type += 1
                self.uav_assists[id].obs_type_times += 1
                self.uav_assists[id].farthest_pos = [None, None]
            else:
                if put is not None and g['s12'] > 400:
                    command = self.put_buoys(id, uav, buoy_all, put=put)
                if command is None or put is None:
                    command = PlaneCommand.move_control(obj_type=RedObjType.UAV, id=id, velocity=650, height=500, course=angle_)


        if self.uav_assists[id].obs_type_times >= len(obs_type):
            self.uav_assists[id].obs_array_finish = True  # 完整一次布阵
        else:
            self.uav_assists[id].obs_array_finish = False

        return command

    def make_decision(self, red_obs: RedGlobalObservation, env_obs: EnvGlobalObservation):
        ####无人机1策略，放置声呐浮标阵列，再巡逻
        # 无人机2策略， 如果阵列没放满先补充完整，再巡逻

        message_infos = {"uav_message": {i: [] for i in range(red_obs.uav_field.uav_nums)},
                         "usv_message": {i: [] for i in range(red_obs.usv_field.usv_nums)}}
        commands = []
        simtime = red_obs.simtime
        self.message = {'uavs': [[] for _ in range(red_obs.uav_field.uav_nums)], 'usvs': [[] for _ in range(red_obs.usv_field.usv_nums)]}

        self.result = {"uav_total_duration_call_point": [0 for _ in range(red_obs.uav_field.uav_nums)],
                           'uav_time_first_identified_sub': 0,
                           "usv_total_duration_call_point": 0,
                           "uav_speed_record": [{"start_time":[], "end_time":[], "describe":[]} for _ in range(red_obs.uav_field.uav_nums)]}

        commands.append(PlaneCommand.Dragg_control(id=0, statu=1))
        commands.append(PlaneCommand.switch_mag(id=0, statu=1))  # 开启磁探传感器
        commands.append(PlaneCommand.switch_infrared(id=0, statu=1))
        commands.append(PlaneCommand.drop_buoy(0, 0, env_obs.cargo_ships[0].lat,  env_obs.cargo_ships[0].lon, -50, 0))
        commands.append(PlaneCommand.drop_buoy(1, 0, env_obs.cargo_ships[0].lat, env_obs.cargo_ships[0].lon, -50, 0))
        commands.append(PlaneCommand.report_target(id=0, target_lat=env_obs.cargo_ships[0].lat, target_lon=env_obs.cargo_ships[0].lon,target_height=-50, target_course=120, target_vel=10, report_time=red_obs.simtime, target_id=0))


        return commands, self.message, message_infos, self.result

    # 平滑转弯
    def smooth_fold(self, cur_angle, tar_angle, d_a):
        if abs(cur_angle - tar_angle) > 1:
            if (90 <= cur_angle and -180 <= tar_angle <= -90) or (
                    0 <= cur_angle and -180 <= tar_angle <= -90 and 180 - abs(cur_angle) + 180 - abs(
                tar_angle) < 180) or (
                    90 <= cur_angle and -90 <= tar_angle <= 0 and 180 - abs(cur_angle) + 180 - abs(tar_angle) < 180):
                if cur_angle >= 180:
                    cur_angle -= 360
                if cur_angle > 0:
                    cur_angle += d_a
                elif cur_angle < 0:
                    cur_angle -= d_a
            elif (cur_angle <= -90 and 90 <= tar_angle <= 180) or (
                    0 <= tar_angle <= 90 and cur_angle <= -90 and 180 - abs(cur_angle) + 180 - abs(
                tar_angle) < 180) or (
                    90 <= tar_angle <= 180 and cur_angle <= 0 and 180 - abs(cur_angle) + 180 - abs(tar_angle) < 180):
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

    # 规避障碍物
    def avoiding_obstacles(self, obj_usv, env_obs):
        tar_angle = obj_usv.course
        res_angle = obj_usv.course
        for boat in env_obs.fishing_boats + env_obs.cargo_ships:
            g_fishing_boat = geod.Inverse(obj_usv.lat, obj_usv.lon, boat.lat, boat.lon)
            if g_fishing_boat['s12'] <= 1000:
                if obj_usv.course > 0 and boat.angle > 0:
                    if 60 > obj_usv.course - boat.angle > 0:
                        tar_angle = obj_usv.course + 10
                    elif 0 > obj_usv.course - boat.angle > -60:
                        tar_angle = obj_usv.course - 10
                elif obj_usv.course > 0 > boat.angle:
                    t = obj_usv.course - 180
                    if 60 > t - boat.angle > 0:
                        tar_angle = obj_usv.course - 10
                    elif 0 > t - boat.angle > -60:
                        tar_angle = obj_usv.course + 10
                elif obj_usv.course < 0 and boat.angle < 0:
                    if -60 < obj_usv.course - boat.angle < 0:
                        tar_angle = obj_usv.course - 10
                    elif 0 < obj_usv.course - boat.angle < 60:
                        tar_angle = obj_usv.course + 10
                elif obj_usv.course < 0 < boat.angle:
                    t = obj_usv.course + 180
                    if -60 < t - boat.angle < 0:
                        tar_angle = obj_usv.course - 10
                    elif 0 < t - boat.angle < 60:
                        tar_angle = obj_usv.course + 10
            res_angle = self.smooth_fold(obj_usv.course, tar_angle, 0.3)
        return res_angle
    def filter_v0(self, vir_pos):
        adc_list = []
        for pos in vir_pos:
            adc_list.append([pos.lat, pos.lon])
        lats = []
        lons = []
        idx0 = 30
        idx1 = 20
        # print(len(adc_list))
        history_dict = {}
        for i, adc_pos in enumerate(adc_list):
            d = []
            for j in range(max(0, i - idx0), i + 1):
                for k in range(j, i + 1):
                    key = str(j) + ',' + str(k)
                    if key in history_dict.keys():
                        d.append(history_dict[key])
                    else:
                        adc0 = adc_list[j]
                        adc1 = adc_list[k]
                        g = geod.Inverse(adc0[0], adc0[1], adc1[0], adc1[1])
                        tmp = [g['s12'], adc0[0], adc0[1], adc1[0], adc1[1]]
                        history_dict[key] = tmp
                        d.append(tmp)
            d.sort(key=lambda x: x[0])
            e = d[int(len(d) * 0.25): int(len(d) * 0.75)]
            if len(e) == 0:
                pass
            else:
                d = e
            # filter d
            mean_lat = np.mean([d[i][1] for i in range(len(d))] + [d[i][3] for i in range(len(d))])
            mean_lon = np.mean([d[i][2] for i in range(len(d))] + [d[i][4] for i in range(len(d))])

            if lats:
                mean_lat = np.mean(lats[-idx1:]) * 0.9 + 0.1 * mean_lat
            if lons:
                mean_lon = np.mean(lons[-idx1:]) * 0.9 + 0.1 * mean_lon

            lats.append(mean_lat)
            lons.append(mean_lon)
        return lats, lons