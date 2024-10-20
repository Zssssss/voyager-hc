import os.path
from multiprocessing import Process
import sys 
# sys.path.append('/home/luzhenning/project/SQ0614')
sys.path.append(r'/')
import yaml
from scene.East_China_Sea import *
geod = Geodesic.WGS84
import argparse
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

task_path = "../config/task.yaml"
################### 读取配置文件 ####################3
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

async def run(websocket):
    mode = 0  # 0：场景导入 1：开始推演
    data_send_flag = True
    time1 = 0
    last_send_time = 0
    done = False

    while True:
        start = time.time()
        try:
            if time1 == 32400 or done:
                await websocket.close()
                return False

            if mode == 0:
                # await websocket.send("start")
                # 数据初始化
                message = await websocket.recv()
                message = json.loads(message)

                if message:
                    global args
                    args = load_Params(args, message)
                    env = EnvTest(args)
                    agent = Decision()
                    mode = 1
                    print("********场景导入成功********")

            if mode == 1:
                result = Result.GO_ON
                if data_send_flag:
                    if time1 == 0:
                        data = env.reset()
                    else:
                        command_dict = agent.make_decision(data)
                        data, result = env.step(command_dict)

                    data.update({"time": time1, "stage": result})
                    if result in [Result.RED_WIN, Result.BLUE_WIN]:
                        done = True

                    last_send_time = time1
                    time1 += 1
                    data_send_flag = False
                    await websocket.send(json.dumps(data, default=lambda o: o.__dict__, sort_keys=True, indent=4))
                message = await websocket.recv()
                # print("运行时间：", time.time() - start)
                if message:  # todo 取消注释
                    # print(message)
                    str_arr = message.strip().split(' ')
                    if len(str_arr) > 2:
                        simtime = int(str_arr[-1])
                        if simtime == last_send_time:
                            data_send_flag = True


        except websockets.ConnectionClosed as e:
            print('connect break')
            mode = 0
            time1 = 0
            last_send_time = 0
            data_send_flag = True
            break

# agent = Decision()

def develop_run(task_id, episode_num, args):
    args = load_Params(args)
    for episode_i in range(episode_num):
        print('#' * 10, task_id, '-', episode_i, '#'*10)
        env = EnvTest(args, task_id, episode_i)
        agent = Decision()
        obs = env.reset()
        done = False
        while not done:
            command_dict = agent.make_decision(obs, task_id, episode_i)
            obs, result = env.step(command_dict)    
            if result in [Result.RED_WIN, Result.BLUE_WIN]:
                done = True
                if result == Result.RED_WIN:
                    print('{}_{} red win'.format(task_id, episode_i))
                else:
                    print('{}_{} sub win'.format(task_id, episode_i))




if __name__ == '__main__':
    if args.prod:
        server_host, server_port = "10.88.115.5", 51001  # todo 取消注释
        # server_host, server_port = "127.0.0.1", 51001  # todo test
        print(f"run with {server_host}:{server_port}...")
        asyncio.get_event_loop().run_until_complete(websockets.serve(run, server_host, server_port))
        asyncio.get_event_loop().run_forever()
    else:
        # 开发者模式 只用于本地开发、训练、测试使用 不需要和前端进行通信
        if not os.path.exists('project/log/'):
            os.makedirs('project/log/')
        task_pool = [Process(target=develop_run, args=(i, 1, args)) for i in range(1)]
        for task in task_pool:
            task.start()
        for task in task_pool:
            task.join()


