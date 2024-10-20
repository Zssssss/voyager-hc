"""
干扰器
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from geographiclib.geodesic import Geodesic
import warnings
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

geod = Geodesic.WGS84


class Jammer_Model:
    def __init__(self):
        self.detect_level = 200  # 探测器声源级,单位dB，一般指的是声呐
        self.noise_fre = 50  # 单位kHz
        self.sea_state = 3
        self.absorption_coefficient = 13  # 吸收系数
        self.target_strength = 20  # 干扰器模拟噪声强度
        self.beam_width = 5
        self.detection_threshold = 20  # 检测阀，越大噪声越大，但是探测方位会变广
        self.fc = 11_000  # 载波频率

    def gen_jammer(self, jammer_lat, jammer_lon, detector_lat, detector_lon, sea_state=1):
        R = geod.Inverse(detector_lat, detector_lon, jammer_lat, jammer_lon)['s12']



        # 回声余量
        A = echo_level(R) - noise_masking_level  # 根据A的大小可以判断是否能对声呐浮标产生干扰，若A＞0表示干扰器对声纳浮标产生了噪声干扰
        if A > 0:
            jammer_flag = True
            A_MAX = echo_level(0.1) - noise_masking_level
            jammer_level = round(A / (A_MAX / len(Qs)))

        return jammer_flag, jammer_level


if __name__ == '__main__':
    import time

    start_time = time.time()
    bait = Jammer_Model()
    save_file_base64, attract = bait.gen_jammer(detector_lat=12, detector_lon=120, jammer_lat=13.01, jammer_lon=120)
    print(save_file_base64, attract)
    end_time = time.time()
    print('total time: {}'.format(end_time - start_time))


