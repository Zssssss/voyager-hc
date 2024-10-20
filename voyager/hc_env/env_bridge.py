import yaml
import argparse
from .warengine.scene.East_China_Sea import *
from typing import SupportsFloat, Any, Tuple, Dict
from gymnasium.core import ObsType
from .warengine.commands.plane_command import *

# import sys
# sys.path.append('/home/ubuntu/zsss/voyager-hc/voyager/hc_env/warengine')

task_path = '/home/ubuntu/zsss/voyager-hc/voyager/hc_env/config/task.yaml'

with open(task_path, encoding='utf-8') as file1:
    args = yaml.load(file1, Loader=yaml.FullLoader)#读取yaml参数文件
parser = argparse.ArgumentParser()
args = argparse.Namespace(**args)
def load_Params(args, message=None):
    # 仿真环境参数
    if message:
        args.Env_params_Start_time = message['env']['time'] if message['env']['time'] in list(str(i) for i in range(1, 4)) else '1'  # 开始时间
        args.Env_params_Sea_state = message['env']['seaState'] if message['env']['seaState'] in list(str(i) for i in range(1, 6)) else '2'  # 海况
        args.Env_params_Inf_num = message['env']['stuff'] if message['env']['stuff'] in list(str(i) for i in range(1, 4)) else '2'  # 干扰物平均数量

        # 潜艇参数
        args.Sub_params_Init_height = message['sub']['depth'] if message['sub']['depth'] in list(str(i) for i in range(1, 4)) else '2'  # 初始深度
        args.Sub_params_Init_Battery = message['sub']['battery'] if message['sub']['battery'] in list(str(i) for i in range(1, 3)) else '1'  # 剩余电量
        args.Sub_params_Init_Bait = message['sub']['bait'] if message['sub']['bait'] in [True, False] else True  # 是否搭载自航式声诱饵
        args.Sub_params_Init_jammer = message['sub']['jammer'] if message['sub']['jammer'] in [True, False] else True  # 是否搭载水声干扰器

        #无人机参数
        args.Uav_params_Init_pos = message['plane']['location'] if message['plane']['location'] in list(str(i) for i in range(1, 3)) else '1'  # 初始位置
        args.Uav_params_Buoys = message['plane']['sonar'] if message['plane']['sonar'] in [True, False] else True  # 是否搭载声呐浮标吊舱
        args.Uav_params_Magnetic = message['plane']['magnetic'] if message['plane']['magnetic'] in [True, False] else True  # 是否搭载磁探仪
        args.Uav_params_Infrared = message['plane']['infrared'] if message['plane']['infrared'] in [True, False] else True  # 是否搭载红外成像仪
        args.Uav_params_MiniSAR = message['plane']['miniSAR'] if message['plane']['miniSAR'] in [True, False] else True  # 是否搭载MiniSAR

        # 无人船参数
        args.Usv_params_Init_pos = message['boat']['location'] if message['boat']['location'] in list(str(i) for i in range(1, 3)) else '2'  # 初始位置
        args.Usv_params_Sonar = message['boat']['sonar'] if message['boat']['sonar'] in [True, False] else True  # 是否搭载拖曳式声纳传感器
    else:
        args.Env_params_Start_time = '1'  # 开始时间
        args.Env_params_Sea_state = '2'  # 海况
        args.Env_params_Inf_num = '2'  # 干扰物平均数量

        # 潜艇参数
        args.Sub_params_Init_height = '2'  # 初始深度
        args.Sub_params_Init_Battery = '1'  # 剩余电量
        args.Sub_params_Init_Bait = True  # 是否搭载自航式声诱饵
        args.Sub_params_Init_jammer = True  # 是否搭载水声干扰器

        # 无人机参数
        args.Uav_params_Init_pos = '1'  # 初始位置
        args.Uav_params_Buoys = True  # 是否搭载声呐浮标吊舱
        args.Uav_params_Magnetic = True  # 是否搭载磁探仪
        args.Uav_params_Infrared = True  # 是否搭载红外成像仪
        args.Uav_params_MiniSAR = True  # 是否搭载MiniSAR

        # 无人船参数
        args.Usv_params_Init_pos = '2'  # 初始位置
        args.Usv_params_Sonar = True  # 是否搭载拖曳式声纳传感器

    return args
args = load_Params(args)



def obj_to_dict(obj):
    """
    Recursively convert objects in a dictionary to dictionaries.
    """
    if isinstance(obj, dict):
        return {key: obj_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [obj_to_dict(item) for item in obj]
    elif hasattr(obj, '__dict__'):  # 检查是否有__dict__属性
        # return {key: obj_to_dict(value) for key, value in obj.__dict__.items() if not key.startswith('__')}  排除私有属性（以双下划线开头的属性）
        return {key: obj_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return obj


class EnvBridge:

    def __init__(self):
        self.env = EnvTest(args, 1, 1)
        self.key_obs = []
        

    def step(self, action_cmds):   
        
        self.pre_obs = self.current_obs

        while len(action_cmds) != 0:
            p = 0
            while "env_forward_t" not in action_cmds[p].keys():
                p += 1
            nxt_obs, nxt_result = self.env.step(action_cmds[:p+1])
            action_cmds = action_cmds[p+1:]
        last_obs, last_result = nxt_obs, nxt_result
        self.key_obs.append(obj_to_dict(last_obs))
        # return last_obs, last_result
        return "generated code execute success"


    def backward(self):
        self.key_obs.pop()
        self.setbystate(self.key_obs[-1])

    def setbystate(self,state_dict):
        return self.env.setbystate(state_dict)

    def reset(self):
        self.env.reset()
        temp_obs, _ = self.env.step({
            "blue_cmds":[],"red_cmds":[],
            "red_message":[],"red_key_message":[],
            "blue_message":[],"blue_key_message":[],
        })
        temp_obs = obj_to_dict(temp_obs)   ####是obs统一
        self.key_obs.append(temp_obs)
        return temp_obs


if __name__ == "__main__":
    env = EnvBridge()
    import pdb;pdb.set_trace()
    print("env init success.")