"""
控制指令：　０　
浮标投放指令：１　
传感器操作：　２
"""


class CmdType:
    MOVE = 1
    PLACE_BUOY = 2
    INFRARED = 3
    MAG = 4
    RADAR = 5
    Jammer = 6
    sub_drag_sonar = 7  # 控制潜艇拖曳声呐
    USV_sonar = 8  # 控制无人艇声呐
    REFILL = 9  # 飞机补充浮标和燃料
    BAIT = 10  # 声诱饵
    REPORT = 11  # 上报目标位置
    sub_sonar = 12  # 控制潜艇舰壳声呐
    Snorkel = 13 #通气管
    Periscope = 14 #潜望镜



class RedObjType:
    UAV = 0
    USV = 1
    PLANE = 2
    SUB = 3
    FRIGATE = 4
    MARITIME = 5



class PlaneCommand:
    @staticmethod
    def move_control(obj_type=RedObjType.UAV, id=0, velocity=None, height=None, course=None):
        return {
            "obj_type": obj_type,
            "id": id,
            "type": CmdType.MOVE,
            'vel': velocity,
            "height": height,
            "course": course
        }


    @staticmethod
    def drop_buoy(buoy_type, channel, lat, lon, height=-175, id=0):
        return {
            "id": id,
            'lat':lat,
            "lon": lon,
            "type": CmdType.PLACE_BUOY,
            "buoy_type": buoy_type,
            "height": height,
            "channel": channel
        }

    # 光电开关
    @staticmethod
    def switch_infrared(id=0, statu=0):
        return {
            "id": id,
            "type": CmdType.INFRARED,
            "statu": statu,
            "obj_type": RedObjType.UAV
        }

    # 上报目标位置
    @staticmethod
    def report_target(id, target_lat, target_lon, target_height, target_course, target_vel,  report_time, obj_type=RedObjType.UAV, target_id=0):
        return {
            "obj_type": obj_type,
            "id": id,
            "type": CmdType.REPORT,
            'target_lat': target_lat,
            "target_lon": target_lon,
            "target_height": target_height,
            "target_course": target_course,
            "target_vel": target_vel,
            "target_id": target_id,
            "report_time": report_time
        }


    # TODO：浮标监听指令

    # TODO：雷达参数控制　

    # 磁探开关
    @staticmethod
    def switch_mag(id=0, statu=0):
        return {
            "id": id,
            "type": CmdType.MAG,
            "statu": statu,
            "obj_type": RedObjType.UAV
        }

    # 无人艇拖拽声呐开关
    @staticmethod
    def Dragg_control(id=0, statu=0, theta_rope=10, rope_len=800, theta_hydrophone=20):
        return {
            "obj_type": RedObjType.USV,
            "id": id,
            "type": CmdType.USV_sonar,
            "statu": statu,
            "theta_rope": theta_rope,
            "rope_len":rope_len,
            "theta_hydrophone":theta_hydrophone

        }
    
    @staticmethod
    def Env_Step():
        return {
            "env_forward_t":1
        }


if __name__ == '__main__':
    commands = []
    commands.append(PlaneCommand.move_control(velocity=450, height=300, course=200))
    print(commands)
