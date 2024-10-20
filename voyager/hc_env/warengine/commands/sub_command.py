
from .plane_command import CmdType

class BlueObjType:
    SUBMARINE = 0
    USV = 1
    PATROL = 2
    DESTROYER = 3
    P1 = 4



class SubmarineCommand:
    @staticmethod
    def move_control(obj_type=BlueObjType.SUBMARINE, id=0, velocity=None, height=None, course=None):
        return {
            "obj_type": obj_type,
            "id": id,
            "type": CmdType.MOVE,
            "vel": velocity,
            "height": height,
            "course": course
        }

    # 声呐
    def drag_sonar_control(obj_type=BlueObjType.SUBMARINE, id=0, statu=0, theta_rope=10, rope_len=800, theta_hydrophone=20):
        #被动拖曳声呐
        """theta_rope:绳缆与海平面夹角
        rope_len:绳缆长度
        theta_hydrophone：拖曳阵与绳缆夹角"""
        return {
            "obj_type": obj_type,
            "statu": statu,
            "id": id,
            "type": CmdType.sub_drag_sonar,
            "theta_rope": theta_rope,
            "rope_len": rope_len,
            "theta_hydrophone": theta_hydrophone
        }

    def sonar_control(obj_type=BlueObjType.SUBMARINE, id=0, statu=0):
        #被动舰壳声呐
        return {
            "obj_type": obj_type,
            "statu": statu,
            "id": id,
            "type": CmdType.sub_sonar,
        }

    # 干扰器
    def jammer_control(obj_type=BlueObjType.SUBMARINE, id=0, height=None, lat=None, lon=None):
        return {
            "obj_type": obj_type,
            "id": id,
            "type": CmdType.Jammer,
            "height": height,
            "lat": lat,
            "lon": lon
        }

    # 声诱饵
    def bait_control(obj_type=BlueObjType.SUBMARINE, id=0, height=None, lat=None, lon=None, velocity=None,
                     course=None):
        return {
            "obj_type": obj_type,
            "id": id,
            "type": CmdType.BAIT,
            "height": height,
            "lat": lat,
            "lon": lon,
            "velocity": velocity,
            "course": course
        }


    # 通气管
    def snorkel_control(obj_type=BlueObjType.SUBMARINE, id=0, statu=0):
        return {
            "obj_type": obj_type,
            "id": id,
            "type": CmdType.Snorkel,
            "statu": statu
        }

    def Periscope_control(obj_type=BlueObjType.SUBMARINE, id=0, statu=0):#潜望镜
        return {
            "obj_type": obj_type,
            "id": id,
            "type": CmdType.Periscope,
            "statu": statu
        }

