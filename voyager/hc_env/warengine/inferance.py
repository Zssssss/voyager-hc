"""
推理平台
"""
from obs.blue_obs import BlueGlobalObservation
from obs.env_obs import EnvGlobalObservation
from obs.red_obs import RedGlobalObservation


class Env:
    def __init__(self,  args):
        self.result_statistics = {}
        # 蓝方仿真结果
        self.sub_type = "基洛级常规潜艇"
        self.sub_pre_pos = [[0, 0] for _ in range(args['submarine_nums'])]  # 潜艇上一时刻位置
        self.sub_low_speed_sailing_duration = [0 for _ in range(args['submarine_nums'])]  # 低速航行时长(速度小于5节)，小时
        self.sub_high_speed_sailing_duration = [0 for _ in range(args['submarine_nums'])]  # 高速航行时长(速度大于6节)，小时
        # TODO: 最后统计的字段
        self.sub_goal_completion_rate = [0 for _ in range(args['submarine_nums'])]  # 目标完成度

        self.sub_total_navigation_mileage = [0 for _ in range(args['submarine_nums'])]  # 航行总里程，米
        self.sub_initial_exposure_time = 0  # 初始暴露时间
        self.sub_bait_deployed_num = 0  # 声诱饵投放数量
        self.sub_jammer_deployed_num = 0  # 干扰器投放数量
        self.sub_velocity_list = [[] for _ in range(args['submarine_nums'])]  # 潜艇速度记录
        self.sub_action_list = [[] for _ in range(args['submarine_nums'])]  # 潜艇动作记录
        self.sub_state_list = [[] for _ in range(args['submarine_nums'])]  # 潜艇状态记录
        # 红方仿真结果
        # 无人机
        self.uav_type = "彩虹-5海洋应用型"
        self.uav_pre_pos = [[0, 0] for _ in range(args['uav_nums'])]  # 无人艇上一时刻位置
        self.uav_total_duration_call_point = [0 for _ in range(args['uav_nums'])]  # 到达应召点总时长，小时
        self.uav_time_first_identified_sub = 0  # 第一次识别到潜艇花费时长，小时
        self.uav_total_navigation_mileage = [0 for _ in range(args['uav_nums'])]  # 航行总里程，米
        self.uav_sonar_buoy_num = {"62": 0, "63": 0, "67": 0, "68": 0, "70": 0}  # 耗费声呐浮标数量
        self.uav_passive_sonar_survival_rate = 0  # 被动声呐存活率
        self.uav_active_sonar_survival_rate = 0  # 主动声呐存活率
        self.uav_target_recognition_accuracy = 0  # 目标识别准确率
        # 无人艇
        self.usv_type = "瞭望者II型无人艇"
        self.usv_pre_pos = [[0, 0] for _ in range(args['usv_nums'])]  # 无人艇上一时刻位置
        self.usv_total_navigation_mileage = [0 for _ in range(args['usv_nums'])]  # 航行总里程，米
        self.usv_total_duration_call_point = 0  # 到达应召点总时长，小时
        self.usv_time_first_identified_sub = 0  # 第一次识别到潜艇花费时长，小时

    def init_obs(self):
        # 环境信息
        obs_message = {
            "red_message": {
                "uav_message": {},
                "usv_message": {}
            },
            "blue_message": {
                "sub_message": {}
            },
            "env_message": {

            }
        }

        blue_obs = BlueGlobalObservation(args) # 蓝方态势数据
        red_obs = RedGlobalObservation(args) #红方态势数据
        env_obs = EnvGlobalObservation(args) # 环境态势数据


        ########### 蓝方态势信息构建 #############
        ########### 红方态势信息构建 ##############
        ########### 环境态势信息构建 ###########

    def reset(self):
        self.red_obs, self.blue_obs, self.env_obs, obs_message = self.init_obs()


    def step(self, commands):
        pass