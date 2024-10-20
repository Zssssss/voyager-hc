import math
import random

import numpy as np
from geographiclib.geodesic import Geodesic

from ..commands.sub_command import SubmarineCommand
from ..obs.env_obs import EnvGlobalObservation
from ..obs.blue_obs import BlueGlobalObservation

geod = Geodesic.WGS84
from ..commands.plane_command import CmdType

red_active_sonar_list = [
    # {"lat": 17.67, "lon": 114.36}, {"lat": 17.65, "lon": 114.52}]
    {"lat": 17.65, "lon": 114.36}, {"lat": 17.70, "lon": 114.50}]


def angle_difference(angle1, angle2):
    """计算两个角度之间的差距。 input angle范围 -180*180"""
    # 将角度转换到0°至360°范围内
    normalized_angle1 = angle1 % 360
    normalized_angle2 = angle2 % 360

    # 计算两个角度之间的差距
    diff = abs(normalized_angle1 - normalized_angle2)

    # 如果差距大于180°，则减去360°得到最短差距
    if diff > 180:
        diff = 360 - diff

    return diff

class SubDecision:
    """
    潜艇策略
    9小时 = 32,400秒

    1节 = 1.852km/h
    1° = 110km
    """

    def __init__(self):
        self.message = []
        self.is_red_sonar = False  # 是否遇到主动声呐
        self.is_avoiding = False  # QT是否正在躲避
        self.is_set_avoid_course = False  # 是否设置QT躲避折线角度
        self.is_set_fold_course = False  # 是否设置QT前进折线角度
        self.is_fold_move = False  # QT是否折线前进
        self.avoid_distance = 5000  # 米，QT躲避主动声呐的距离（QT距离主动声呐多远时开始折线躲避）
        self.stop_avoid_distance = 10000  # QT停止躲避主动声呐的距离（QT躲避主动声呐多远时停止躲避）
        self.first_meet_red_sonar_distance = 2000  # 第一次遇到红方主动声呐的距离
        self.blue_sonar_distance = 10000  # QT声呐探测距离
        self.fold_time = 0  # 遇到主动声呐后拐弯的时间
        self.fold_course = 0  # 遇到主动声呐后拐弯的角度
        self.fold_flag = 1  # 拐弯时刻记录
        self.cur_course = 90  # QT当前航向
        self.is_jammer = False  # 是否投放干扰器
        self.is_bait = False  # 是否投放声诱饵
        self.bait_mode = -1  # 声诱饵投放模式。0：初始一个角度，然后一定时间后再向QT目的地出发；1：初始向主动声呐前进，然后一定时间后再向QT目的地出发
        self.mode = 0 #0为搜索模式
        self.detect_sonar_sus_info = []#记录舰壳声呐探测到的可疑信息
        self.detect_drag_sonar_sus_info = []  # 记录拖曳声呐探测到的可疑信息
        self.detect_up_sus_info = {"drag_sonar": [], "sonar": []}  # 跃变层上方的可疑目标，去掉了潜艇和鱼群
        self.detect_down_sus_info = {"drag_sonar": [], "sonar": []}  # 跃变层下方的可疑目标，去掉了潜艇和鱼群
        self.red_agent_info = {"sub":[], "usv":[],"uav": [], "buoy":[]}# 记录探测到红方智能体的信息, 只是获取最近的目标
        self.sub_course = None #记录潜艇原航向
        self.avoid_start_time = None #记录开始躲避的时间
        self.turn_course = None #记录目标转向的角度
        self.bait_course = None #记录投放声诱饵的航向
        self.sub_height = None #记录潜艇的高度
        self.buoy = None #记录红方浮标信息（决策时候的特定浮标信息）
        self.usv_info = None #记录潜艇浮标信息（决策时候的特定浮标信息）
        self.uav_info = None  # 记录潜艇浮标信息（决策时候的特定浮标信息）
        self.long_dis_turn_course = None#记录长距离转向的航向
        self.buoy_time = None #记录self.buoy开始生效的事件
        self.usv_time = None  # 记录self.usv_info开始生效的事件
        self.uav_time = None  # 记录self.usv_info开始生效的事件
        self.bait_drop = False # 记录是否需要投放声诱饵
        self.buoy_avoid_success = False
        self.upward = [0, False]
        self.red_sub_track_times = [0, False]
        self.key_message = []
        self.sub_track_report_time = 0
        self.avoid_report_time = 0
        self.mode0_new =  False #是否不是第一次经历mode0
        self.sub_track_times = 0  # 发现并跟踪红方潜艇时长 单位s
        self.large_course_times = 0  # 大角度规避次数
        self.statistics_result = {}
        self.avoid_time_record = None

    def make_decision(self, blue_obs: BlueGlobalObservation):
        self.message = []
        self.key_message = []
        ###仅针对一个潜艇目标
        id = 0
        commands = []
        obj = blue_obs.submarines[id]
        search_area = obj.sub_field #潜艇任务海域
        if blue_obs.simtime == 0:
            self.search_mode = [0 for _ in range(blue_obs.submarine_nums)]#搜索模式的形式状态
            self.start_lat = [0 for _ in range(blue_obs.submarine_nums)]
            self.start_lon = [0 for _ in range(blue_obs.submarine_nums)]
            self.seach_mode_arrive = [True for _ in range(blue_obs.submarine_nums)]#seach_mode 刚刚转换模式

            self.patrol_search_mode = [0 for _ in range(blue_obs.patrol_ship_nums)]
            self.patrol_start_lat = [0 for _ in range(blue_obs.patrol_ship_nums)]
            self.patrol_start_lon = [0 for _ in range(blue_obs.patrol_ship_nums)]
            self.patrol_seach_mode_arrive = [True for _ in range(blue_obs.patrol_ship_nums)]  # seach_mode 刚刚转换模式

            self.destroyer_search_mode = [0 for _ in range(blue_obs.destroyer_nums)]
            self.destroyer_start_lat = [0 for _ in range(blue_obs.destroyer_nums)]
            self.destroyer_start_lon = [0 for _ in range(blue_obs.destroyer_nums)]
            self.destroyer_seach_mode_arrive = [True for _ in range(blue_obs.destroyer_nums)]

            self.P1_pos = [[[],[]] for _ in range(blue_obs.P1_plane_nums)]
            self.P1_pos_index = [0 for _ in range(blue_obs.P1_plane_nums)]
            self.P1_angle_plan = [True for _ in range(blue_obs.P1_plane_nums)]
            self.P1_centor_pos = [[0,0] for _ in range(blue_obs.P1_plane_nums)]
            self.P1_patrol_start_pos = [[] for _ in range(blue_obs.P1_plane_nums)]
            self.P1_staright_start = [True for _ in range(blue_obs.P1_plane_nums)]
            self.P1_staright_index = [0 for _ in range(blue_obs.P1_plane_nums)]


        # ############################### easy version ################################
        # target_point = {
        #     "lat": search_area['min_lat'],
        #     "lon": search_area['min_lon']
        # }
        #
        # g = geod.Inverse(obj.lat, obj.lon, target_point['lat'], target_point['lon'])
        # commands.append(SubmarineCommand.move_control(id=id, velocity=8, height=-50, course=g['azi1']))
        # return commands, self.message

        self.detect_sonar_sus_info = []  # 记录舰壳声呐探测到的可疑信息
        self.detect_drag_sonar_sus_info = []  # 记录拖曳声呐探测到的可疑信息
        self.detect_up_sus_info = {"drag_sonar": [], "sonar": []}  # 跃变层上方的可疑目标，去掉了潜艇和鱼群
        self.detect_down_sus_info = {"drag_sonar": [], "sonar": []}  # 跃变层下方的可疑目标，去掉了潜艇和鱼群
        # self.truncate_list(blue_obs.simtime, list_=self.detect_sonar_sus_info)
        # self.truncate_list(blue_obs.simtime, list_=self.detect_drag_sonar_sus_info)
        # self.truncate_list(blue_obs.simtime, list_=self.detect_up_sus_info["drag_sonar"])
        # self.truncate_list(blue_obs.simtime, list_=self.detect_up_sus_info["sonar"])
        # self.truncate_list(blue_obs.simtime, list_=self.detect_down_sus_info["drag_sonar"])
        # self.truncate_list(blue_obs.simtime, list_=self.detect_down_sus_info["sonar"])
        self.truncate_list(blue_obs.simtime, list_=self.red_agent_info["sub"], list_class=0)
        self.truncate_list(blue_obs.simtime, list_=self.red_agent_info["usv"], list_class=0)
        self.truncate_list(blue_obs.simtime, list_=self.red_agent_info["uav"], list_class=0)
        self.truncate_list(blue_obs.simtime, list_=self.red_agent_info["buoy"], list_class=0)



        if self.mode == 0:
            if self.avoid_time_record is not None:
                self.avoid_time_record += 1

            if obj.drag_sonar.target_pos is None and obj.sonar.target_course is None:
                commands.append(SubmarineCommand.sonar_control(id=id, statu=1))  # 打开舰壳声呐， 搜索跃变层上方
                commands.append(SubmarineCommand.drag_sonar_control(id=id, statu=1, theta_rope=20, rope_len=140,
                                                                    theta_hydrophone=20))  # 拖曳声呐在跃变层下方
            else: #探测到可疑目标
                self.sensor_data(blue_obs, obj) # 更新探测信息
                    #追踪红方潜艇
                if len(self.red_agent_info['buoy']) > 0:  # 跃变层以下还有可疑目标应该是主动声呐，率先躲避
                    # print('发现红方主动浮标，进行躲避')
                    self.mode = 1  # 躲避主动声呐
                    # message = {"title": "潜艇发现红方主动声呐浮标", "detail": f"{id+1}号潜艇发现红方{len(self.red_agent_info['buoy'][-1][f'{blue_obs.simtime}'])}个主动声呐浮标", 'time': blue_obs.simtime}
                    # self.message.append(message) if message not in self.message else None
                    # message = {"info": "潜艇发现红方主动声呐浮标", "class": 'blue', "type": 'submarine',"id": id}
                    # self.key_message.append(message) if message not in self.key_message else None
                elif len(self.red_agent_info['usv']) > 0:
                    # print('发现红方无人艇，进行躲避')
                    self.mode = 3  # 躲避无人艇
                    if not self.mode0_new:
                        message = {"title": "潜艇发现红方无人艇", "detail": f"{id + 1}号潜艇发现红方{len(self.red_agent_info['usv'][-1][f'{blue_obs.simtime}'])}艘无人艇", 'time': blue_obs.simtime}
                        self.message.append(message) if message not in self.message else None
                        # message = {"info": "潜艇发现红方无人艇", "class": 'blue', "type": 'submarine', "id": id}
                        # self.key_message.append(message) if message not in self.key_message else None
                elif len(self.detect_up_sus_info["drag_sonar"]) > 0 or len(self.detect_up_sus_info["sonar"]) > 0:# 跃变层以上有可疑目标
                    if self.avoid_time_record is None or self.avoid_time_record > 10*60:
                        if self.upward[0] == 0 or (blue_obs.simtime-self.upward[0]) > 20*60: #20分钟不会上浮
                            self.upward[1] = True
                            self.upward[0] = blue_obs.simtime
                            message = {"title": "潜艇发现跃变层以上的可疑目标", "detail": f"{id + 1}号潜艇发现跃变层以上存在可疑目标", 'time': blue_obs.simtime}
                            self.message.append(message) if message not in self.message else None
                            message = {"info": "潜艇发现跃变层以上的可疑目标", "class": 'blue', "type": 'submarine', "id": id}
                            self.key_message.append(message) if message not in self.key_message else None
                    else:
                        self.upward[1] = False

                if self.upward[1]: #上浮
                    course = None
                    if len(self.red_agent_info['sub']) > 0:
                        # print('发现红方潜艇，进行跟踪')
                        key = list(self.red_agent_info['sub'][-1].keys())[0]
                        if blue_obs.simtime - int(key) < 1 * 60:
                            sub = self.red_agent_info['sub'][-1]
                            if "course" not in list(self.red_agent_info['sub'][-1][key][-1].keys()):
                                g = geod.Inverse(obj.lat, obj.lon, sub[key][-1]['lat'], sub[key][-1]['lon'])
                                course = g['azi1']
                            else:
                                course = sub[key][-1]['course']

                    commands.append(SubmarineCommand.move_control(id=id, velocity=4, height=-10, course=course))  # 探测到水上有可疑目标，上浮并打开潜望镜，同时进行充电
                    commands.append(SubmarineCommand.drag_sonar_control(id=id, statu=1, theta_rope=20, rope_len=np.clip(140, 4 * obj.height + 340, 300), theta_hydrophone=20))  # 时刻控制拖曳声呐在跃变层下方
                    if obj.height >= -10:
                        commands.append(SubmarineCommand.snorkel_control(id=id, statu=1))  # 打开通气管
                        commands.append(SubmarineCommand.Periscope_control(id=id, statu=1))  # 打开潜望镜
                        commands.append(SubmarineCommand.drag_sonar_control(id=id, statu=1, theta_rope=20, rope_len=300, theta_hydrophone=20))  # 拖曳声呐在跃变层下方
                        if obj.periscope.touch:
                            self.periscope_data(blue_obs, obj)
                            if len(self.red_agent_info['uav']) > 0:
                                # print('发现红方无人机，进行躲避')
                                self.mode = 2  # 躲避无人机
                            elif len(self.red_agent_info['usv']) > 0:
                                # print('发现红方无人艇，进行躲避')
                                self.mode = 3  # 躲避无人艇
                            elif len(self.red_agent_info['buoy']) > 0:
                                # print('发现红方主动浮标，进行躲避')
                                self.mode = 1
                            self.upward[1] = False
                        else:
                            if obj.battery.battery_ratio > 0.7:
                                self.upward[1] = False #不需要充电

                else:
                    if len(self.red_agent_info['sub']) > 0:
                        # print('发现红方潜艇，进行跟踪')
                        if self.sub_track_report_time != blue_obs.simtime or self.sub_track_report_time == 0:
                            self.sub_track_report_time = blue_obs.simtime
                            message = {"title": "潜艇发现红方潜艇", "detail": f"{id + 1}号潜艇发现红方潜艇", 'time': blue_obs.simtime}
                            self.message.append(message) if message not in self.message else None
                            message = {"info": "潜艇发现红方潜艇", "class": 'blue', "type": 'submarine', "id": id}
                            self.key_message.append(message) if message not in self.key_message else None
                        self.sub_track_report_time += 1

                        key = list(self.red_agent_info['sub'][-1].keys())[0]
                        if not self.mode0_new:
                            avoid_time = 1 * 60
                            sub_height = -50
                        else:
                            avoid_time = 5 * 60 #15min 不上浮
                            sub_height = -110
                        if self.avoid_time_record is not None and self.avoid_time_record >= 10 * 60:
                            sub_height = -50
                        if blue_obs.simtime - int(key) < avoid_time: #没有信息的时候 avoid_time时间内还是根据原先的探测信息
                            sub = self.red_agent_info['sub'][-1]
                            self.sub_track_times += 1
                            if "course" not in list(self.red_agent_info['sub'][-1][key][-1].keys()):
                                g = geod.Inverse(obj.lat, obj.lon, sub[key][-1]['lat'], sub[key][-1]['lon'])
                                if g['s12'] > blue_obs.blue_sonar - 2000:
                                    commands.append(SubmarineCommand.move_control(id=id, velocity=8, height=sub_height, course=g['azi1']))
                                else:
                                    commands.append(SubmarineCommand.move_control(id=id, velocity=4, height=sub_height, course=g['azi1']))
                            else:
                                commands.append(SubmarineCommand.move_control(id=id, velocity=6, height=sub_height, course=sub[key][-1]['course']))




            for command in commands:
                # move = False  # 没有机动动作的时候采用下面的检查方法
                if command['type'] == CmdType.MOVE:
                    break
                if command == commands[-1]:
                    # move = True
                    # 检查搜索
                    self.search_mode[id] = self.search_mode[id] % 4
                    if self.search_mode[id] == 0:  # 从左到右
                        if self.seach_mode_arrive[id]:
                            self.target_lat = obj.lat
                            self.target_lon = search_area["max_lon"]
                            self.seach_mode_arrive[id] = False
                        course = 90
                    elif self.search_mode[id] == 1 or self.search_mode[id] == 3:  # 从上到下
                        if self.seach_mode_arrive[id]:
                            g = geod.Direct(obj.lat, obj.lon, 180, blue_obs.blue_sonar)
                            self.target_lat = max(g['lat2'], search_area["min_lat"])
                            self.target_lon = obj.lon
                            self.seach_mode_arrive[id] = False
                        course = 180
                    else:  # 从右到左
                        if self.seach_mode_arrive[id]:
                            self.target_lat = obj.lat
                            self.target_lon = search_area["min_lon"]
                            self.seach_mode_arrive[id] = False
                        course = -90
                    g = geod.Inverse(obj.lat, obj.lon, self.target_lat, self.target_lon)
                    if g['s12'] < 5000:
                        self.search_mode[id] += 1
                        self.seach_mode_arrive[id] = True
                    else:
                        if not self.mode0_new:
                            sub_height = -50
                        else:
                            sub_height = -80
                        if self.avoid_time_record is not None and self.avoid_time_record >= 10 * 60:
                            sub_height = -50
                        commands.append(SubmarineCommand.move_control(id=id, velocity=6, height=sub_height, course=course))
            if self.avoid_time_record is not None and self.avoid_time_record < 10 * 60:
                self.mode = 0




        if self.mode == 1: #躲避主动声呐
            if self.sub_course is None:
                self.sub_course = obj.course

            if self.buoy_time is not None: #长时间没有摆脱
                if (abs(blue_obs.simtime - self.buoy_time) > 10*60 and int(list(self.red_agent_info['buoy'][-1].keys())[0]) > self.buoy_time and self.avoid_start_time is None) or self.red_sub_track_times[1]:
                    # print('长时间没有摆脱主动浮标,根据当前探测信息重新规划')
                    self.turn_course = None
                    self.bait_course = None
                    self.buoy = None
                    self.long_dis_turn_course = None
                    self.sub_height = None
                    self.avoid_report_time = 0

            if self.buoy is None:
                buoys = self.red_agent_info['buoy'][-1]
                key = list(buoys.keys())[0]
                self.buoy_time = int(key)
                # 选择影响最大的声呐浮标躲避
                dis = np.inf
                drag_sonar_buoy = None
                for now_buoy in buoys[key]:
                    if "course" not in list(now_buoy.keys()):
                        g = geod.Inverse(obj.lat, obj.lon, now_buoy["lat"], now_buoy['lon'])
                        if dis > g['s12']:#距离最近的
                            dis = g['s12']
                            drag_sonar_buoy = now_buoy

                p = -np.inf
                sonar_buoy = None
                for now_buoy in buoys[key]:
                    if "course" in list(now_buoy.keys()):
                        p_ = now_buoy['p']
                        if p < p_: #噪声信号最强的
                            p = p_
                            sonar_buoy = now_buoy
                if sonar_buoy is None:
                    self.buoy = drag_sonar_buoy
                else:
                    if drag_sonar_buoy is None:
                        self.buoy = sonar_buoy
                    else:
                        if dis >= blue_obs.blue_sonar:
                            self.buoy = sonar_buoy
                        else:
                            self.buoy = drag_sonar_buoy


                if "course" not in list(self.buoy.keys()):
                    g = geod.Inverse(obj.lat, obj.lon, self.buoy["lat"], self.buoy['lon'])
                    if g['s12'] > 5000: #假设主动声呐浮标探测范围为5000m
                        if self.long_dis_turn_course is None:
                            # print('在主动声呐浮标探测范围之外，反向规避')
                            self.long_dis_turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_lat=self.buoy['lat'], target_lon=self.buoy['lon'], turn_course=180, target_info="pos")
                            message = {"title": "潜艇规避主动声呐浮标", "detail": f"{id + 1}号潜艇大角度转弯规避主动声呐浮标，规避角度为{self.long_dis_turn_course}", 'time': blue_obs.simtime}
                            self.message.append(message) if message not in self.message else None
                            self.large_course_times += 1
                            if self.avoid_report_time == 0:
                                message = {"info": "潜艇规避主动声呐浮标", "class": 'blue', "type": 'submarine', "id": id}
                                self.key_message.append(message) if message not in self.key_message else None
                                self.avoid_report_time += 1
                        commands.append(SubmarineCommand.move_control(id=id, velocity=10, height=-50, course=self.long_dis_turn_course))#背向声呐浮标
                        if self.sub_height is None:
                            self.sub_height = -50
                    else:
                        if self.turn_course is None:
                            # print('在主动声呐浮标探测范围之内')
                            self.turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_lat=self.buoy['lat'], target_lon=self.buoy['lon'], target_info="pos", target_difference=30)

                            message = {"title": "潜艇规避主动声呐浮标", "detail": f"{id + 1}号潜艇小角度转弯规避主动声呐浮标，规避角度为{self.turn_course}", 'time': blue_obs.simtime}
                            self.message.append(message) if message not in self.message else None
                            if self.avoid_report_time == 0:
                                message = {"info": "潜艇规避主动声呐浮标", "class": 'blue', "type": 'submarine', "id": id}
                                self.key_message.append(message) if message not in self.key_message else None
                                self.avoid_report_time += 1

                        if self.bait_course is None:
                            if len(blue_obs.submarines[id].bait) <= blue_obs.submarines[id].bait_nums:
                                self.bait_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_lat=self.buoy['lat'], target_lon=self.buoy['lon'], target_info="pos", turn_course=120, target="near", agent="bait")  # 靠近声呐投放声诱饵
                                self.bait_drop = True

                else:
                    if self.buoy['p'] < 30:
                        if self.long_dis_turn_course is None:
                            # print('在主动声呐浮标探测范围之外，反向规避')
                            self.long_dis_turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_course=self.buoy['course'], turn_course=180, target_info="course")
                            message = {"title": "潜艇规避主动声呐浮标", "detail": f"{id + 1}号潜艇大角度转弯规避主动声呐浮标，规避角度为{self.long_dis_turn_course}", 'time': blue_obs.simtime}
                            self.message.append(message) if message not in self.message else None
                            self.large_course_times += 1
                            if self.avoid_report_time == 0:
                                message = {"info": "潜艇规避主动声呐浮标", "class": 'blue', "type": 'submarine', "id": id}
                                self.key_message.append(message) if message not in self.key_message else None
                                self.avoid_report_time += 1
                        commands.append(SubmarineCommand.move_control(id=id, velocity=10, height=-50, course=self.long_dis_turn_course))  # 背向声呐浮标
                    else:
                        if self.turn_course is None:
                            # print('在主动声呐浮标探测范围之内')
                            self.turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_course=self.buoy['course'], target_info="course", target_difference=30)
                            message = {"title": "潜艇规避主动声呐浮标", "detail": f"{id + 1}号潜艇小角度转弯规避主动声呐浮标，规避角度为{self.turn_course}", 'time': blue_obs.simtime}
                            self.message.append(message) if message not in self.message else None
                            if self.avoid_report_time == 0:
                                message = {"info": "潜艇规避主动声呐浮标", "class": 'blue', "type": 'submarine', "id": id}
                                self.key_message.append(message) if message not in self.key_message else None
                                self.avoid_report_time += 1
                        if self.bait_course is None:
                            if len(blue_obs.submarines[id].bait) <= blue_obs.submarines[id].bait_nums:
                                self.bait_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_course=self.buoy['course'], target_info="course", turn_course=120, target="near", agent="bait")  # 靠近声呐投放声诱饵
                                self.bait_drop = True



                if self.turn_course is not None:
                    commands.append(SubmarineCommand.move_control(id=id, velocity=10, height=blue_obs.thermocline - 90, course=self.turn_course))  # 背向声呐浮标
                    if self.sub_height is None:
                        self.sub_height = blue_obs.thermocline - 90

                if self.bait_course is not None:
                    if len(blue_obs.submarines[id].bait) == 0:
                        commands.append(SubmarineCommand.bait_control(id=id, height=obj.height, lat=obj.lat, lon=obj.lon, velocity=10, course=self.bait_course))
                        self.bait_drop = False
                    else:
                        if blue_obs.simtime - blue_obs.submarines[id].bait[-1].start_time > 10 * 60:  # 10分钟只能投放一个
                            if self.bait_drop:
                                commands.append(SubmarineCommand.bait_control(id=id, height=obj.height, lat=obj.lat, lon=obj.lon, velocity=10, course=self.bait_course))
                                self.bait_drop = False
                            self.bait_course = None

                if len(blue_obs.submarines[id].jammers) <= blue_obs.submarines[id].jammer_nums:
                    if "course" not in list(self.buoy.keys()):
                        g = geod.Inverse(obj.lat, obj.lon, obj.course, target_lat=self.buoy['lat'],  target_lon=self.buoy['lon'])
                        buoy_course = g['azi1']
                    else:
                        buoy_course = self.buoy['course']
                    if angle_difference(obj.course, buoy_course) < 10:  # 靠近声呐浮标的时候投放干扰器
                        if len(blue_obs.submarines[id].jammers) == 0:
                            commands.append(SubmarineCommand.jammer_control(id=id, height=obj.height, lat=obj.lat, lon=obj.lon))
                        else:
                            if blue_obs.simtime - blue_obs.submarines[id].jammers[-1].start_time > 10 * 60:  # 10分钟只能投放一个
                                commands.append(SubmarineCommand.jammer_control(id=id, height=obj.height, lat=obj.lat, lon=obj.lon))

                self.sensor_data(blue_obs, obj)#探测传感器结果更新

                if len(self.red_agent_info['buoy']) == 0:#浮标没有探测信息
                    self.buoy_avoid_success = True
                else:
                    if int(list(self.red_agent_info['buoy'][-1].keys())[0]) < blue_obs.simtime:
                        self.buoy_avoid_success = True
                    else:
                        self.buoy_avoid_success = False

                if self.buoy_avoid_success:#浮标没有探测信息
                    if self.avoid_start_time is None:
                        self.avoid_start_time = blue_obs.simtime
                    else:
                        self.avoid_start_time += 1

                    if self.avoid_start_time > 5 * 60:
                        message = {"title": "潜艇摆脱主动声呐浮标", "detail": f"{id + 1}号潜艇连续5分钟摆脱主动声呐浮标监测", 'time': blue_obs.simtime}
                        self.message.append(message) if message not in self.message else None
                        message = {"info": "潜艇摆脱主动声呐浮标", "class": 'blue', "type": 'submarine', "id": id}
                        self.key_message.append(message) if message not in self.key_message else None
                        self.mode = 0
                        course = obj.course % 360
                        self.seach_mode_arrive[id] = True
                        if course <= 180 and course >= 0:
                            g = geod.Inverse(obj.lat, obj.lon, obj.lat, search_area['max_lon'])
                            if g['s12'] > 5000:
                                self.search_mode[id] = 0
                            else:
                                self.search_mode[id] = 1
                        else:
                            g = geod.Inverse(obj.lat, obj.lon, obj.lat, search_area['min_lon'])
                            if g['s12'] > 5000:
                                self.search_mode[id] = 2
                            else:
                                self.search_mode[id] = 3
                        commands.append(SubmarineCommand.move_control(id=id, velocity=6, height=-80, course=self.sub_course))  # 恢复航向
                        self.sub_course = None
                        self.avoid_start_time = None
                        self.turn_course = None
                        self.bait_course = None
                        self.buoy = None
                        self.sub_height = None
                        self.long_dis_turn_course = None
                        self.buoy_time = None
                        self.bait_drop = False
                        self.buoy_avoid_success = False
                        self.upward[1] = False
                        self.red_agent_info = {"sub": [], "usv": [], "uav": [], "buoy": []}
                        self.red_sub_track_times = [0, False]
                        self.sub_track_report_time = 0
                        self.avoid_report_time = 0
                        self.mode0_new = True
                        self.avoid_time_record = 0


        if self.mode == 2:#躲避无人机
            if self.sub_course is None:
                self.sub_course = obj.course
            if self.uav_time is not None: #长时间没有摆脱
                if (abs(blue_obs.simtime - self.uav_time) > 5*60 and int(list(self.red_agent_info['uav'][-1].keys())[0]) > self.uav_time and self.avoid_start_time is None) or self.red_sub_track_times[1]:
                    self.turn_course = None
                    self.uav_info = None
                    self.long_dis_turn_course = None
                    self.sub_height = None
                    self.avoid_report_time = 0
            if self.uav_info is None:
                uavs = self.red_agent_info['uav'][-1]
                key = list(uavs.keys())[0]
                dis = np.inf
                uav = None
                for now_uav in uavs[key]:
                    g = geod.Inverse(obj.lat, obj.lon, now_uav["lat"], now_uav['lon'])
                    if dis > g['s12']:  # 距离最近的
                        dis = g['s12']
                        uav = now_uav
                self.uav_info = uav

            if self.turn_course is None:
                self.turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_lat=self.uav_info["lat"], target_lon=self.uav_info['lon'], target_info="pos", turn_course=100, target="flee", target_difference=30)
                message = {"title": "潜艇规避无人机", "detail": f"{id + 1}号潜艇快速下潜以规避无人机", 'time': blue_obs.simtime}
                self.message.append(message) if message not in self.message else None
                if self.avoid_report_time == 0:
                    message = {"info": "潜艇规避无人机", "class": 'blue', "type": 'submarine', "id": id}
                    self.key_message.append(message) if message not in self.key_message else None
                    self.avoid_report_time += 1
            commands.append(SubmarineCommand.move_control(id=id, velocity=10, height=-110, course=self.turn_course))

            if abs(obj.height + 110) < 10:
                if self.avoid_start_time is None:
                    self.avoid_start_time = blue_obs.simtime
                if blue_obs.simtime - self.avoid_start_time > 5 * 60:
                    message = {"title": "潜艇摆脱无人机", "detail": f"{id + 1}号潜艇下潜5分钟摆脱无人机监测", 'time': blue_obs.simtime}
                    self.message.append(message) if message not in self.message else None
                    message = {"info": "潜艇摆脱无人机", "class": 'blue', "type": 'submarine', "id": id}
                    self.key_message.append(message) if message not in self.key_message else None
                    self.mode = 0
                    course = obj.course % 360
                    self.seach_mode_arrive[id] = True
                    if course <= 180 and course >= 0:
                        g = geod.Inverse(obj.lat, obj.lon, obj.lat, search_area['max_lon'])
                        if g['s12'] > 5000:
                            self.search_mode[id] = 0
                        else:
                            self.search_mode[id] = 1
                    else:
                        g = geod.Inverse(obj.lat, obj.lon, obj.lat, search_area['min_lon'])
                        if g['s12'] > 5000:
                            self.search_mode[id] = 2
                        else:
                            self.search_mode[id] = 3
                    commands.append(SubmarineCommand.move_control(id=id, velocity=6, height=-80, course=self.sub_course))  # 恢复航向
                    self.sub_course = None
                    self.avoid_start_time = None
                    self.turn_course = None
                    self.upward[1] = False
                    self.red_agent_info = {"sub": [], "usv": [], "uav": [], "buoy": []}
                    self.red_sub_track_times = [0, False]
                    self.uav_info = None
                    self.uav_time = None
                    self.avoid_report_time = 0
                    self.sub_track_report_time = 0
                    self.mode0_new = True
                    self.avoid_time_record = 0

        if self.mode == 3:#躲避无人艇
            if self.sub_course is None:
                self.sub_course = obj.course
            if self.usv_time is not None: #长时间没有摆脱
                if (abs(blue_obs.simtime - self.usv_time) > 5*60 and int(list(self.red_agent_info['usv'][-1].keys())[0]) > self.usv_time and self.avoid_start_time is None) or self.red_sub_track_times[1]:
                    self.turn_course = None
                    self.usv_info = None
                    self.long_dis_turn_course = None
                    self.sub_height = None
                    self.avoid_report_time = 0
            if self.usv_info is None:
                usvs = self.red_agent_info['usv'][-1]
                key = list(usvs.keys())[0]
                self.usv_time = int(key)
                # 选择影响最大的声呐浮标躲避
                dis = np.inf
                pos_usv = None
                for now_usv in usvs[key]:
                    if "course" not in list(now_usv.keys()):
                        g = geod.Inverse(obj.lat, obj.lon, now_usv["lat"], now_usv['lon'])
                        if dis > g['s12']:#距离最近的
                            dis = g['s12']
                            pos_usv = now_usv

                p = -np.inf
                course_usv = None
                for now_usv in usvs[key]:
                    if "course" in list(now_usv.keys()):
                        p_ = now_usv['p']
                        if p < p_: #噪声信号最强的
                            p = p_
                            course_usv = now_usv
                if course_usv is None:
                    self.usv_info = pos_usv
                else:
                    if pos_usv is None:
                        self.usv_info = course_usv
                    else:
                        if dis >= blue_obs.blue_sonar:
                            self.usv_info = course_usv
                        else:
                            self.usv_info = pos_usv


            if "course" not in list(self.usv_info.keys()):
                g = geod.Inverse(obj.lat, obj.lon, self.usv_info["lat"], self.usv_info['lon'])
                if g['s12'] > 8000: #假设无人艇声呐探测范围为8000m
                    if self.long_dis_turn_course is None:
                        self.long_dis_turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_lat=self.usv_info['lat'], target_lon=self.usv_info['lon'], turn_course=30, target_info="pos")
                        message = {"title": "潜艇规避无人艇", "detail": f"{id + 1}号潜艇大角度转弯规避无人艇，规避角度为{self.long_dis_turn_course}", 'time': blue_obs.simtime}
                        self.message.append(message) if message not in self.message else None
                        self.large_course_times += 1
                        if self.avoid_report_time == 0:
                            message = {"info": "潜艇规避无人艇", "class": 'blue', "type": 'submarine', "id": id}
                            self.key_message.append(message) if message not in self.key_message else None
                            self.avoid_report_time += 1
                    commands.append(SubmarineCommand.move_control(id=id, velocity=10, height=-50, course=self.long_dis_turn_course))
                    if self.sub_height is None:
                        self.sub_height = -50
                else:
                    if self.turn_course is None:
                        self.turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_lat=self.usv_info['lat'], target_lon=self.usv_info['lon'], turn_course=110, target_info="pos", target_difference=30)
                        message = {"title": "潜艇规避无人艇", "detail": f"{id + 1}号潜艇小角度转弯规避无人艇，规避角度为{self.turn_course}", 'time': blue_obs.simtime}
                        self.message.append(message) if message not in self.message else None
                        if self.avoid_report_time == 0:
                            message = {"info": "潜艇规避无人艇", "class": 'blue', "type": 'submarine', "id": id}
                            self.key_message.append(message) if message not in self.key_message else None
                            self.avoid_report_time += 1

            else:
                if self.usv_info['p'] < 30:
                    if self.long_dis_turn_course is None:
                        self.long_dis_turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_lat=self.usv_info['lat'], target_lon=self.usv_info['lon'], turn_course=30, target_info="pos")
                        message = {"title": "潜艇规避无人艇", "detail": f"{id + 1}号潜艇大角度转弯规避无人艇，规避角度为{self.long_dis_turn_course}", 'time': blue_obs.simtime}
                        self.message.append(message) if message not in self.message else None
                        self.large_course_times += 1
                        if self.avoid_report_time == 0:
                            message = {"info": "潜艇规避无人艇", "class": 'blue', "type": 'submarine', "id": id}
                            self.key_message.append(message) if message not in self.key_message else None
                            self.avoid_report_time += 1
                    commands.append(SubmarineCommand.move_control(id=id, velocity=10, height=-50, course=self.long_dis_turn_course))
                else:
                    if self.turn_course is None:
                        self.turn_course = self.cal_turn_course(obj.lat, obj.lon, obj.course, target_course=self.usv_info['course'], turn_course=110, target_info="course", target_difference=30)
                        message = {"title": "潜艇规避无人艇", "detail": f"{id + 1}号潜艇小角度转弯规避无人艇，规避角度为{self.turn_course}", 'time': blue_obs.simtime}
                        self.message.append(message) if message not in self.message else None
                        if self.avoid_report_time == 0:
                            message = {"info": "潜艇规避无人艇", "class": 'blue', "type": 'submarine', "id": id}
                            self.key_message.append(message) if message not in self.key_message else None
                            self.avoid_report_time += 1

            if self.turn_course is not None:
                commands.append(SubmarineCommand.move_control(id=id, velocity=10, height=blue_obs.thermocline - 90, course=self.turn_course))  # 背向声呐浮标
                if self.sub_height is None:
                    self.sub_height = blue_obs.thermocline - 90

            self.sensor_data(blue_obs, obj)#探测传感器结果更新

            if len(self.red_agent_info['usv']) == 0:
                self.buoy_avoid_success = True
            else:
                if int(list(self.red_agent_info['usv'][-1].keys())[0]) < blue_obs.simtime:
                    self.buoy_avoid_success = True
                else:
                    self.buoy_avoid_success = False

            if self.buoy_avoid_success:
                if self.avoid_start_time is None:
                    self.avoid_start_time = blue_obs.simtime

                if blue_obs.simtime - self.avoid_start_time > 5 * 60:
                    message = {"title": "潜艇摆脱无人艇", "detail": f"{id + 1}号潜艇连续5分钟摆脱主动声呐浮标监测", 'time': blue_obs.simtime}
                    self.message.append(message) if message not in self.message else None
                    message = {"info": "潜艇摆脱无人艇", "class": 'blue', "type": 'submarine', "id": id}
                    self.key_message.append(message) if message not in self.key_message else None
                    self.mode = 0
                    course = obj.course % 360
                    self.seach_mode_arrive[id] = True
                    if course <= 180 and course >= 0:
                        g = geod.Inverse(obj.lat, obj.lon, obj.lat, search_area['max_lon'])
                        if g['s12'] > 5000:
                            self.search_mode[id] = 0
                        else:
                            self.search_mode[id] = 1
                    else:
                        g = geod.Inverse(obj.lat, obj.lon, obj.lat, search_area['min_lon'])
                        if g['s12'] > 5000:
                            self.search_mode[id] = 2
                        else:
                            self.search_mode[id] = 3
                    commands.append(SubmarineCommand.move_control(id=id, velocity=6, height=-80, course=self.sub_course))  # 恢复航向
                    self.sub_course = None
                    self.avoid_start_time = None
                    self.turn_course = None
                    self.usv_info = None
                    self.usv_time = None
                    self.buoy_avoid_success = False
                    self.upward[1] = False
                    self.red_agent_info = {"sub": [], "usv": [], "uav": [], "buoy": []}
                    self.long_dis_turn_course = None
                    self.sub_height = None
                    self.red_sub_track_times = [0, False]
                    self.sub_track_report_time = 0
                    self.avoid_report_time = 0
                    self.mode0_new = True
                    self.avoid_time_record = 0





        id = 1 #苍龙级潜艇，巡逻一下
        obj = blue_obs.submarines[id]
        partrol_rad = 60 * 1.852 * 1000 #搜索海域的边长 60 海里
        self.search_mode[id] = self.search_mode[id] % 4
        if self.search_mode[id] == 0:  # 从左到右
            if self.seach_mode_arrive[id]:
                self.start_lat[id] = obj.lat
                self.start_lon[id] = obj.lon
                self.seach_mode_arrive[id] = False
            partrol_dis = partrol_rad
            course = 90
        elif self.search_mode[id] == 1 or self.search_mode[id] == 3:  # 从上到下
            if self.seach_mode_arrive[id]:
                self.start_lat[id] = obj.lat
                self.start_lon[id] = obj.lon
                self.seach_mode_arrive[id] = False
            partrol_dis = blue_obs.blue_sonar
            course = 180
        else:  # 从右到左
            if self.seach_mode_arrive[id]:
                self.start_lat[id] = obj.lat
                self.start_lon[id] = obj.lon
                self.seach_mode_arrive[id] = False
            partrol_dis = partrol_rad
            course = -90
        g = geod.Inverse(self.start_lat[id], self.start_lon[id], obj.lat, obj.lon)
        if g['s12'] > partrol_dis:
            self.search_mode[id] += 1
            self.seach_mode_arrive[id] = True
        else:
            commands.append(SubmarineCommand.move_control(id=id, velocity=6, height=-50, course=course))


        # 均轻级巡逻舰控制
        for id in range(blue_obs.patrol_ship_nums):
            obj = blue_obs.patrol_ships[id]
            partrol_rad = 40 * 1.852 * 1000  #搜索海域的边长 40 海里
            self.patrol_search_mode[id] = self.patrol_search_mode[id] % 4
            if self.patrol_search_mode[id] == 0:  # 从左到右
                if self.patrol_seach_mode_arrive[id]:
                    self.patrol_start_lat[id] = obj.lat
                    self.patrol_start_lon[id] = obj.lon
                    self.patrol_seach_mode_arrive[id] = False
                partrol_dis = partrol_rad
                course = 90
            elif self.patrol_search_mode[id] == 1 or self.patrol_search_mode[id] == 3:  # 从上到下
                if self.patrol_seach_mode_arrive[id]:
                    self.patrol_start_lat[id] = obj.lat
                    self.patrol_start_lon[id] = obj.lon
                    self.patrol_seach_mode_arrive[id] = False
                partrol_dis = 6_000
                course = 180
            else:  # 从右到左
                if self.patrol_seach_mode_arrive[id]:
                    self.patrol_start_lat[id] = obj.lat
                    self.patrol_start_lon[id] = obj.lon
                    self.patrol_seach_mode_arrive[id] = False
                partrol_dis = partrol_rad
                course = -90
            g = geod.Inverse(self.patrol_start_lat[id], self.patrol_start_lon[id], obj.lat, obj.lon)
            if g['s12'] > partrol_dis:
                self.patrol_search_mode[id] += 1
                self.patrol_seach_mode_arrive[id] = True
            else:
                commands.append(SubmarineCommand.move_control(obj_type=2, id=id, velocity=20, height=0, course=course))


        # 驱逐舰控制
        for id in range(blue_obs.destroyer_nums):
            obj = blue_obs.destroyers[id]
            partrol_rad = 12 * 1.852 * 1000  # 搜索海域的边长 12 海里
            self.destroyer_search_mode[id] = self.destroyer_search_mode[id] % 4
            if self.destroyer_search_mode[id] == 0:  # 从左到右
                if self.destroyer_seach_mode_arrive[id]:
                    self.destroyer_start_lat[id] = obj.lat
                    self.destroyer_start_lon[id] = obj.lon
                    self.destroyer_seach_mode_arrive[id] = False
                partrol_dis = partrol_rad
                course = 90
            elif self.destroyer_search_mode[id] == 1 or self.destroyer_search_mode[id] == 3:  # 从上到下
                if self.destroyer_seach_mode_arrive[id]:
                    self.destroyer_start_lat[id] = obj.lat
                    self.destroyer_start_lon[id] = obj.lon
                    self.destroyer_seach_mode_arrive[id] = False
                partrol_dis = 3_000
                course = 180
            else:  # 从右到左
                if self.destroyer_seach_mode_arrive[id]:
                    self.destroyer_start_lat[id] = obj.lat
                    self.destroyer_start_lon[id] = obj.lon
                    self.destroyer_seach_mode_arrive[id] = False
                partrol_dis = partrol_rad
                course = -90
            g = geod.Inverse(self.destroyer_start_lat[id], self.destroyer_start_lon[id], obj.lat, obj.lon)
            if g['s12'] > partrol_dis:
                self.destroyer_search_mode[id] += 1
                self.destroyer_seach_mode_arrive[id] = True
            else:
                commands.append(
                    SubmarineCommand.move_control(obj_type=3, id=id, velocity=30, height=0, course=course))


        #p-1飞机控制
        for id in range(blue_obs.P1_plane_nums):
            obj = blue_obs.P1_plane[id]
            if self.P1_angle_plan[id]:
                g = geod.Direct(obj.lat, obj.lon, azi1=np.random.uniform(30, 60), s12=np.random.uniform(12_000, 15_000))
                self.P1_centor_pos[id][0], self.P1_centor_pos[id][1] = g['lat2'], g['lon2']
                g = geod.Inverse(self.P1_centor_pos[id][0], self.P1_centor_pos[id][1], obj.lat, obj.lon)
                for angle in np.arange(g['azi1'], 360 + g['azi1'], 360 / 12):
                    g_ = geod.Direct(lat1=self.P1_centor_pos[id][0], lon1=self.P1_centor_pos[id][1],
                                     s12=g["s12"], azi1=angle)
                    self.P1_patrol_start_pos[id].append([g_['lat2'], g_['lon2']])
                    self.P1_angle_plan[id] = False  # 起始点规划完毕

            if not self.P1_angle_plan[id] and self.P1_staright_start[id]:
                self.P1_staright_index[id] = self.P1_staright_index[id] % len(self.P1_patrol_start_pos[id])
                g = geod.Inverse(obj.lat, obj.lon, self.P1_patrol_start_pos[id][self.P1_staright_index[id]][0],self.P1_patrol_start_pos[id][self.P1_staright_index[id]][1])
                if g['s12'] < 1000:
                    self.P1_staright_start[id] = False
                    self.P1_staright_index[id] += 1
                else:
                    commands.append(SubmarineCommand.move_control(obj_type=4, id=id, velocity=800, height=800, course=g['azi1']))

            if not self.P1_angle_plan[id] and not self.P1_staright_start[id]:
                if len(self.P1_pos[id][0]) == 0:
                    self.P1_pos[id][0], self.P1_pos[id][1] = self.patrol_path(obj.lat, obj.lon, self.P1_centor_pos[id][0], self.P1_centor_pos[id][1], 5_000, num=30)

                if len(self.P1_pos[id][0]) > 0:
                    g = geod.Inverse(obj.lat, obj.lon,  self.P1_pos[id][0][self.P1_pos_index[id]], self.P1_pos[id][1][self.P1_pos_index[id]])
                    if g['s12'] < 1000:
                        self.P1_pos_index[id] += 1
                        if self.P1_pos_index[id] == len(self.P1_pos[id][0]):
                            self.P1_pos[id] = [[], []]
                            self.P1_staright_start[id] = True
                            self.P1_pos_index[id] = 0
                    else:
                        commands.append(SubmarineCommand.move_control(obj_type=4, id=id, velocity=800, height=800, course=g['azi1']))

        self.statistics_result={'sub_track_times': self.sub_track_times,"large_course_times":self.large_course_times}


        return commands, self.message, self.key_message, self.statistics_result

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


    def sensor_data(self, blue_obs, obj):
        if obj.drag_sonar.target_pos is not None:
            self.detect_drag_sonar_sus_info.append({"simtime": blue_obs.simtime, "self_pos": {"lat": obj.sonar.self_pos['lat'], "lon": obj.sonar.self_pos['lon'], "height": obj.sonar.self_pos['height'], "sonar_height": np.mean([obj.drag_sonar.hydrophone_pos[f'sonar{i}']['height'] \
                                                     for i in range( obj.drag_sonar.hydrophone_num)])}, "target_pos": obj.drag_sonar.target_pos, "target_feature": obj.drag_sonar.target_feature[ 'sonar0']})  # [{"simtime":0, 'target_pos': [{'lat': xx, 'lon': xx}], 'target_feature': [{'theta': xx, 'f': xx, 'p_recevied': xx}]}]
            # 去除鱼群信息
            target = self.detect_drag_sonar_sus_info[-1]
            index = [i for i in range(len(target['target_feature'])) if target['target_feature'][i]['f'] >= 150]
            target['target_pos'] = [target['target_pos'][i] for i in range(len(target['target_pos'])) if i in index]
            target['target_feature'] = [f for f in target['target_feature'] if f['f'] >= 150]
            if len(target['target_feature']) == 0:
                self.detect_drag_sonar_sus_info.remove(target)

        if obj.sonar.target_course is not None:
            self.detect_sonar_sus_info.append({"simtime": blue_obs.simtime, "self_pos": obj.sonar.self_pos, "target_course": obj.sonar.target_course, "target_feature": obj.sonar.target_feature})  # [{'4': {'target_course': [xx, xx], 'target_feature': [{'theta': xx, 'f': xx, 'p_recevied':xx}, {'theta': xx, 'f': xx 'p_recevied': xx}]}}]
            # 去除鱼群信息
            target = self.detect_sonar_sus_info[-1]
            index = [i for i in range(len(target['target_feature'])) if target['target_feature'][i]['f'] >= 150]
            target['target_course'] = [target['target_course'][i] for i in range(len(target['target_course'])) if i in index]
            target['target_feature'] = [f for f in target['target_feature'] if f['f'] >= 150]
            if len(target['target_feature']) == 0:
                self.detect_sonar_sus_info.remove(target)

        if len(self.detect_drag_sonar_sus_info) > 0 or len(self.detect_sonar_sus_info) > 0:
            if len(self.detect_drag_sonar_sus_info) > 0:
                target = self.detect_drag_sonar_sus_info[-1]
                height = target['self_pos']["sonar_height"]
                # 处理潜艇探测信息
                index = [i for i in range(len(target['target_feature'])) if target['target_feature'][i]['f'] >= 1000]
                if len(index) > 0:
                    self.red_agent_info['sub'].append({f"{blue_obs.simtime}": [{"lat": target['target_pos'][i]['lat'], "lon": target['target_pos'][i]['lon'], "sonar_height": height} for i in index]})
                    target['target_pos'] = [target['target_pos'][i] for i in range(len(target['target_pos'])) if i not in index]
                    target['target_feature'] = [f for f in target['target_feature'] if f['f'] < 1000]
                    if len(target['target_feature']) == 0:
                        self.detect_drag_sonar_sus_info.remove(target)

                if len(target['target_feature']) > 0:
                    if height >= blue_obs.thermocline:  # 探测到的信息在跃变层上方
                        self.detect_up_sus_info["drag_sonar"].append({"simtime": blue_obs.simtime, "target_pos": target['target_pos'], "target_feature": target['target_feature'], "sonar_height": height})
                    else:
                        self.detect_down_sus_info["drag_sonar"].append({"simtime": blue_obs.simtime, "target_pos": target['target_pos'], "target_feature": target['target_feature'], "sonar_height": height})

            if len(self.detect_sonar_sus_info) > 0:
                target = self.detect_sonar_sus_info[-1]
                height = target['self_pos']['height']
                # 处理潜艇探测信息
                index = [i for i in range(len(target['target_feature'])) if target['target_feature'][i]['f'] >= 1000]
                if len(index) > 0:
                    if len(self.red_agent_info['sub']) > 0:
                        if int(list(self.red_agent_info['sub'][-1].keys())[0]) == blue_obs.simtime:
                            info = self.red_agent_info['sub'][-1][f"{blue_obs.simtime}"]
                            if "course" not in list(info[-1].keys()):
                                for i in index:
                                    append = True
                                    for sub in info:
                                        g = geod.Inverse(target['self_pos']['lat'], target['self_pos']['lon'], sub['lat'], sub['lon'])
                                        if abs(g['azi1'] - target['target_course'][i]) > 5 or g['s12'] > blue_obs.blue_sonar:  # 认为不是一个目标
                                            continue
                                        else:
                                            append = False
                                            break
                                    if append:
                                        self.red_agent_info['sub'][-1][f"{blue_obs.simtime}"].extend([{"self_pos": target['self_pos'], "course": target['target_course'][i]}])

                        else:
                            self.red_agent_info['sub'].append({f"{blue_obs.simtime}": [{"self_pos": target['self_pos'], "course": target['target_course'][i]} for i in index]})
                    else:
                        self.red_agent_info['sub'].append({f"{blue_obs.simtime}": [{"self_pos": target['self_pos'], "course": target['target_course'][i]} for i in index]})

                    target['target_course'] = [target['target_course'][i] for i in range(len(target['target_course'])) if i not in index]
                    target['target_feature'] = [f for f in target['target_feature'] if f['f'] < 1000]  # 去除红方潜艇目标
                    if len(target['target_feature']) == 0:
                        self.detect_sonar_sus_info.remove(target)

                if len(target['target_feature']) > 0:
                    if height >= blue_obs.thermocline:  # 探测到的信息在跃变层上方
                        self.detect_up_sus_info["sonar"].append({"simtime": blue_obs.simtime, "self_pos": target['self_pos'], "target_course": target['target_course'], "target_feature": target['target_feature']})
                    else:
                        self.detect_down_sus_info["sonar"].append({"simtime": blue_obs.simtime, "self_pos": target['self_pos'], "target_course": target['target_course'], "target_feature": target['target_feature']})

            # 跃变层以下还有可疑目标应该是主动声呐，率先躲避
            if len(self.detect_down_sus_info['drag_sonar']) > 0 or len(self.detect_down_sus_info['sonar']) > 0:
                if len(self.detect_down_sus_info['drag_sonar']) > 0:
                    sus = self.detect_down_sus_info['drag_sonar'][-1]
                    self.red_agent_info['buoy'].append({f"{blue_obs.simtime}": [{"lat": sus['target_pos'][i]['lat'], "lon": sus['target_pos'][i]['lon'], "sonar_height": sus["sonar_height"]} for i in range(len(sus['target_pos']))]})

                if len(self.detect_down_sus_info['sonar']) > 0:
                    sus = self.detect_down_sus_info['sonar'][-1]
                    if len(self.red_agent_info['buoy']) > 0:
                        if int(list(self.red_agent_info['buoy'][-1].keys())[0]) == blue_obs.simtime:
                            info = self.red_agent_info['buoy'][-1][f"{blue_obs.simtime}"]
                            if "course" not in list(info[-1].keys()):
                                for i in range(len(sus['target_course'])):
                                    target_course = sus['target_course'][i]
                                    append = True
                                    for buoy in info:
                                        g = geod.Inverse(sus['self_pos']['lat'], sus['self_pos']['lon'], buoy['lat'], buoy['lon'])
                                        if abs(g['azi1'] - target_course) > 5 or g['s12'] > blue_obs.blue_sonar:  # 认为不是一个目标
                                            continue
                                        else:
                                            append = False
                                            break
                                    if append:
                                        self.red_agent_info['buoy'][-1][f"{blue_obs.simtime}"].extend([{"self_pos": sus['self_pos'], "course": target_course, 'p': sus['target_feature'][i]['p_recevied']}])
                        else:
                            self.red_agent_info['buoy'].append({f"{blue_obs.simtime}": [{"self_pos": sus['self_pos'], "course": sus['target_course'][i], 'p': sus['target_feature'][i]['p_recevied']} for i in range(len(sus['target_course']))]})
                    else:
                        self.red_agent_info['buoy'].append({f"{blue_obs.simtime}": [{"self_pos": sus['self_pos'], "course": sus['target_course'][i], 'p': sus['target_feature'][i]['p_recevied']} for i in range(len(sus['target_course']))]})

            elif len(self.detect_up_sus_info["drag_sonar"]) > 0 or len(self.detect_up_sus_info["sonar"]) > 0:
                # 跃变层以上有可疑目标，先判断是否是红方usv
                if len(self.detect_up_sus_info["drag_sonar"]) > 0:
                    sus = self.detect_up_sus_info["drag_sonar'"][-1]
                    index = [i for i in range(len(sus['target_feature'])) if sus['target_feature'][i]['f'] > 400]
                    self.red_agent_info['usv'].append({f"{blue_obs.simtime}": [{"lat": sus['target_pos'][i]['lat'], "lon": sus['target_pos'][i]['lon'], "sonar_height": sus["sonar_height"]} for i in index]})

                if len(self.detect_up_sus_info["sonar"]) > 0:
                    sus = self.detect_up_sus_info["sonar"][-1]
                    for i in range(len(sus['target_course'])):
                        if sus['target_feature'][i]['f'] > 400:  # 存在无人艇
                            if len(self.red_agent_info['usv']) > 0:
                                if len(self.red_agent_info['usv'][-1]) == 0:  # 第一个时刻
                                    self.red_agent_info['usv'].append({f"{blue_obs.simtime}": [{"self_pos": sus['self_pos'], "course": sus['target_course'][i], 'p': sus['target_feature'][i]['p_recevied']}]})
                                else:
                                    if int(list(self.red_agent_info['usv'][-1].keys())[-1]) == blue_obs.simtime:  # 一个时刻有多个无人艇，或者是拖曳声呐探测到的目标
                                        append = True
                                        for info in self.red_agent_info['usv'][-1][f"{blue_obs.simtime}"]:
                                            if "course" not in list(info.keys()):  # 探测信息是来自于拖曳声呐
                                                g = geod.Inverse(sus['self_pos']['lat'], sus['self_pos']['lon'], info['lat'], info['lon'])
                                                if abs(g['azi1'] - sus['target_course'][i]) > 5 or g['s12'] > blue_obs.blue_sonar:  # 认为不是一个目标
                                                    continue
                                                else:
                                                    append = False
                                                    break
                                        if append:
                                            self.red_agent_info['usv'][-1][f"{blue_obs.simtime}"].extend([{"self_pos": sus['self_pos'], "course": sus['target_course'][i], 'p': sus['target_feature'][i]['p_recevied']}])
                                    else:
                                        self.red_agent_info['usv'].append({f"{blue_obs.simtime}": [{"self_pos": sus['self_pos'], "course": sus['target_course'][i], 'p': sus['target_feature'][i]['p_recevied']}]})
                            else:
                                self.red_agent_info['usv'].append({f"{blue_obs.simtime}": [{"self_pos": sus['self_pos'], "course": sus['target_course'][i], 'p': sus['target_feature'][i]['p_recevied']}]})

        if len(self.red_agent_info['sub']) > 0: # 每追踪红方潜艇5min，要更换目标位置信息
            if int(list(self.red_agent_info['sub'][-1].keys())[0]) == blue_obs.simtime:
                if self.red_sub_track_times[0] % (5*60) == 0:
                    self.red_sub_track_times[1] = True
                else:
                    self.red_sub_track_times[1] = False
                self.red_sub_track_times[0] += 1
            else:
                self.red_sub_track_times[1] = False


    def periscope_data(self, blue_obs, obj):
        env_index = {"drag_sonar": [], "sonar": []}
        for periscope_result in obj.periscope.result:
            if list(periscope_result.keys())[0] == "uav":
                if len(self.red_agent_info['uav']) == 0:
                    self.red_agent_info['uav'].append({f"{blue_obs.simtime}": [periscope_result['uav']]})
                else:
                    if blue_obs.simtime == int(list(self.red_agent_info['uav'][-1].keys())[0]):  # 看到多个无人机目标
                        self.red_agent_info['uav'][-1][f"{blue_obs.simtime}"].extend([periscope_result['uav']])  # self.red_agent_info['uav']根据时间顺序排列，最后一个就是最新时间有两个uav目标时，self.red_agent_info['uav'][-1] = {'19': [{'lat': xx, 'lon': xx, 'height': xx}, {'lat': xx, 'lon': xx, 'height': xx}]}
                    else:
                        self.red_agent_info['uav'].append({f"{blue_obs.simtime}": [periscope_result['uav']]})

            elif list(periscope_result.keys())[0] == "usv":
                if len(self.red_agent_info['usv']) == 0:
                    self.red_agent_info['usv'].append({f"{blue_obs.simtime}": [periscope_result['usv']]})
                else:
                    if blue_obs.simtime == int(list(self.red_agent_info['usv'][-1].keys())[0]):
                        self.red_agent_info['usv'][-1][f"{blue_obs.simtime}"].extend([periscope_result['usv']])  # self.red_agent_info['usv']根据时间顺序排列，最后一个就是最新时间有两个usv目标时，self.red_agent_info['usv'][-1] = {'19': [{'lat': xx, 'lon': xx, 'height': xx}, {'lat': xx, 'lon': xx, 'height': xx}]}
                    else:
                        self.red_agent_info['usv'].append({f"{blue_obs.simtime}": [periscope_result['usv']]})


            else:
                # 发现跃变层上方的主动声呐-- 排除跃变层上方的渔船、货轮、usv（此时可以认为声呐没有探测到usv、潜望镜没有探测到usv、uav）
                key = list(periscope_result.keys())[0]
                if len(self.detect_up_sus_info['drag_sonar']) > 0:
                    sus = self.detect_up_sus_info['drag_sonar'][-1]
                    # 潜望镜探测到的信息渔船、货轮没有和拖曳声呐探测信息重叠
                    env_index["drag_sonar"] += [i for i in range(len(sus['target_feature'])) if geod.Inverse(periscope_result[key]['lat'], periscope_result[key]['lon'], sus['target_pos'][i]['lat'], sus['target_pos'][i]['lon'])['s12'] <= 800]
                if len(self.detect_up_sus_info['sonar']) > 0:
                    sus = self.detect_up_sus_info['sonar'][-1]
                    # 潜望镜探测到的信息渔船、货轮没有和拖曳声呐探测信息重叠
                    g = geod.Inverse(obj.lat, obj.lon, periscope_result[key]['lat'], periscope_result[key]['lon'])
                    env_index["sonar"] += [i for i in range(len(sus['target_feature'])) if (abs(g['azi1'] - sus['target_course'][i]) <= 5 or g['s12'] <= blue_obs.blue_sonar)]

        if len(env_index["drag_sonar"]) > 0:
            sus = self.detect_up_sus_info['drag_sonar'][-1]
            index = [i for i in range(len(sus['target_feature'])) if i not in env_index["drag_sonar"]]
            self.red_agent_info['buoy'].append({f"{blue_obs.simtime}": [{"lat": sus['target_pos'][i]['lat'], "lon": sus['target_pos'][i]['lon'], "sonar_height": sus["sonar_height"]} for i in index]})

        if len(env_index["sonar"]) > 0:
            sus = self.detect_up_sus_info['sonar'][-1]
            index = [i for i in range(len(sus['target_feature'])) if i not in env_index["sonar"]]
            if len(self.red_agent_info['buoy']) > 0:
                if int(list(self.red_agent_info['buoy'][-1].keys())[0]) == blue_obs.simtime:
                    info = self.red_agent_info['buoy'][-1][f"{blue_obs.simtime}"]
                    if "course" not in list(info[-1].keys()):
                        for i in range(len(sus['target_course'])):
                            target_course = sus['target_course'][i]
                            append = True
                            for buoy in info:
                                g = geod.Inverse(obj.lat, obj.lon, buoy['lat'], buoy['lon'])
                                if abs(g['azi1'] - target_course) > 5 or g['s12'] > blue_obs.blue_sonar:  # 认为不是一个目标
                                    continue
                                else:
                                    append = False
                                    break
                            if append:
                                self.red_agent_info['buoy'][-1][f"{blue_obs.simtime}"].extend([{"self_pos": sus[
                                    'self_pos'], "course": target_course, 'p': sus['target_feature'][i][
                                    'p_recevied']}])
                else:
                    self.red_agent_info['buoy'].append({f"{blue_obs.simtime}": [
                        {"self_pos": sus['self_pos'], "course": sus['target_course'][i],
                         'p': sus['target_feature'][i]['p_recevied']} for i in range(len(sus['target_course']))]})
            else:
                self.red_agent_info['buoy'].append({f"{blue_obs.simtime}": [
                    {"self_pos": sus['self_pos'], "course": sus['target_course'][i],
                     'p': sus['target_feature'][i]['p_recevied']} for i in range(len(sus['target_course']))]})

    def truncate_list(self, simtime, list_, list_class=1, max_length=10):
        updata_time = 8 * 60
        if len(list_) > max_length:
            del list_[:(len(list_) - max_length)]
        if list_class == 1:
            if len(list_) > 0:
                if abs(simtime - list_[-1]["simtime"]) > updata_time:
                    list_ = [] #很久没有有用的数据
        else:
            if len(list_) > 0:
                if abs(simtime - int(list(list_[-1].keys())[0])) > updata_time:
                    list_ =[]
        return list_

    def cal_rotation(self, angle1, angle2, target_difference=50):
        if angle2 is None:
            return angle1
        normalized_angle1 = angle1 % 360
        normalized_angle2 = angle2 % 360

        # 计算两个角度之间的差距
        diff = abs(normalized_angle1 - normalized_angle2)

        # 如果差距大于180°，则减去360°得到最短差距
        if diff > 180:
            diff = 360 - diff
        if diff <= target_difference:
            return angle1
        else:
            if angle_difference(normalized_angle2 + target_difference, normalized_angle1) < angle_difference(
                    normalized_angle2 - target_difference, normalized_angle1):
                angel = normalized_angle2 + target_difference
            else:
                angel = normalized_angle2 - target_difference
        angel = (angel + 180) % 360 - 180
        return angel

    def cal_turn_course(self, lat, lon, course, target_lat=None, target_lon=None, target_course=None, target_info="course", turn_course=90, target="flee", agent="sub", target_difference=50):
            """根据探测到信息控制角度"""
            turn_sub_course = None
            if len(self.red_agent_info['sub'])>0 and agent=='sub':
                sub = self.red_agent_info['sub'][-1]
                index = list(sub.keys())[0]
                if "course" not in list(sub[index][-1].keys()):
                    turn_sub_course = geod.Inverse(lat, lon, sub[index][-1]['lat'], sub[index][-1]['lon'])['azi1']
                else:
                    turn_sub_course = sub[index][-1]['course']

            if target_info == "course": #只知道目标的角度信息
                if target == "flee":#逃离目标
                    if angle_difference(course - turn_course, target_course) < angle_difference(course + turn_course, target_course):
                        if turn_sub_course is not None:
                            return self.cal_rotation(course + turn_course, turn_sub_course, target_difference)
                    else:
                        if turn_sub_course is not None:
                            return self.cal_rotation(course - turn_course, turn_sub_course, target_difference)
                else:
                    if angle_difference(course - turn_course, target_course) > angle_difference(course + turn_course, target_course):
                        return self.cal_rotation(course + turn_course, turn_sub_course, target_difference)
                    else:
                        return self.cal_rotation(course - turn_course, turn_sub_course, target_difference)
            else:#已知目标的位置信息
                g = geod.Inverse(lat, lon, target_lat, target_lon)
                if target == "flee":  # 逃离为目标
                    if angle_difference(course - turn_course, g['azi1']) < angle_difference(course + turn_course, g['azi1']):#向左转90度距离目标更近
                        return self.cal_rotation(course + turn_course, turn_sub_course, target_difference)
                    else:
                        return self.cal_rotation(course - turn_course, turn_sub_course, target_difference)
                else:
                    if angle_difference(course - turn_course, g['azi1']) > angle_difference(course + turn_course, g['azi1']):
                        return self.cal_rotation(course + turn_course, turn_sub_course, target_difference)
                    else:
                        return self.cal_rotation(course - turn_course, turn_sub_course, target_difference)






if __name__ == '__main__':
    s = geod.Inverse(17.77, 114.29, red_active_sonar_list[0]["lat"], red_active_sonar_list[0]["lon"])
    print(s)

    # g = geod.Direct(lat, lon,
    #                 s12=vel * 1.83 * 1000 / 3600,
    #                 azi1=course)