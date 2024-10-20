"""
雷达建模
"""
import base64
import math
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt

from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84
"""
提高功率 可以增加信噪比 可以提高识别qt的精度

"""


class RadarModel:
    def __init__(self):
        cs = []

    def cal_pd(self, dis, Ss):
        pfa = 1e-8
        pt = 1.5e6
        freq = 5.6e9
        c = 3e8
        lamb = c / freq  # 波长
        p_peak = 10 * math.log10(pt)  # 峰值功率转化为DB形式
        lamb_sqdb = 10 * math.log10(lamb ** 2)
        sigma = 0.1  # 雷达目标截面积
        simgdb = 10 * math.log10(sigma)
        four_pi_cub = 10 * math.log10((4 * math.pi) ** 3)
        te = 290
        te_db = 10 * math.log10(te)
        b = 5e6  # 雷达带宽
        b_db = 10 * math.log10(b)
        if Ss < 3:
            k = 2
        else:
            k = 10
        dis /= (2 ** (-(Ss - 3) / (2 * k)))
        range_pwr4_db = 10 * math.log10(dis ** 4)
        k_db = 10 * math.log10(1.38e-23)
        g = 45  # 天线增益
        num = p_peak + g + g + lamb_sqdb + simgdb
        nf = 3  # 噪声系数  3db
        loss = 6  # 雷达损失 6db
        den = four_pi_cub + k_db + te_db + b_db + nf + loss + range_pwr4_db
        snr = num - den
        pd = cal_pd(snr=snr, pfa=pfa)
        return pd

    def gen_radar_img(self, plane_lat, plane_lon, sub_lat, sub_lon, Ss=3, sensor_img=False):
        g = geod.Inverse(plane_lat, plane_lon, sub_lat, sub_lon)
        dis = g["s12"]
        pd = self.cal_pd(dis=dis, Ss=Ss)
        u = np.random.rand()

        point_num = 720
        angles = np.linspace(0, 360, point_num)
        touch = False
        course = None
        if u <= pd:
            touch = True
            r = normality(angles, dis=1000, p=pd, deg=g["azi1"] % 360) + [np.random.random() for _ in range(point_num)]
            r /= max(r)
            course = np.where(r == max(r))[0][0] / (point_num / 360)
        else:
            touch = False
            r = [np.random.random() * 0.01 for _ in range(point_num)]
        thetas = np.linspace(0, np.pi * 2, point_num)
        if touch and sensor_img:
            fig = plt.figure()
            plt.rcParams['axes.facecolor'] = '#343541'
            plt.rcParams['font.size'] = 10  # 字体大小
            plt.rcParams['xtick.color'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
            # print(plt.rcParams)
            fig.patch.set_facecolor('#343541')
            fig.patch.set_alpha(0.9)
            ax = plt.subplot(projection='polar')
            # width = np.pi /  * np.random.rand()
            width = np.pi * 2 / point_num * 8
            colors = plt.get_cmap('autumn')(r)
            # plt.ylim((0, 0.1))
            ax.bar(thetas, r, width=width, color=colors, alpha=0.5)
            # plt.savefig('.\\radar.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
            # plt.title('radar')
            # plt.show()

            save_file = BytesIO()
            plt.savefig(save_file, format="png")
            save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
        else:
            save_file_base64 = None
        # plt.clf()
        # plt.close()
        return save_file_base64, touch, course


def marcumsq_parl(a, b):
    max_test_value = 1000

    if a < b:
        alphan0 = 1
        dn = a / b
    else:
        alphan0 = 0
        dn = b / a

    alphan_1 = 0
    betan0 = 0.5
    betan_1 = 0
    D1 = dn
    n = 0
    ratio = 2 / (a * b)
    rl = 0
    alphan = 0
    betan = 0

    while betan < max_test_value:
        n = n + 1
        alphan = dn + ratio * n * alphan0 + alphan
        betan = 1.0 + ratio * n * betan0 + betan
        alphan_1 = alphan0
        alphan0 = alphan
        betan_1 = betan0
        betan0 = betan
        dn = dn * D1

    Pd = (alphan0 / (2.0 * betan0)) * math.exp(-(a - b) ** 2 / 2.0)

    if a >= b:
        Pd = 1.0 - Pd
    return Pd


def cal_pd(pfa, snr):
    b = math.sqrt(-2 * math.log(pfa))
    a = math.sqrt(2 * 10 ** (snr / 10))
    pd = marcumsq_parl(a, b)
    return pd


def normality(x, deg, dis, p):
    # print(p)
    mean = deg
    std = 1 / (p * 0.1)
    A = p * 10 * (np.sqrt(2 * math.pi) * std)
    return A / (np.sqrt(2 * math.pi) * std) * np.exp(-(x - mean) ** 2 / (2 * std * std))


if __name__ == '__main__':
    import time

    start_time = time.time()
    radar = RadarModel()
    img, touch, course = radar.gen_radar_img(plane_lat=15, plane_lon=120, sub_lat=12.001, sub_lon=120, Ss=3,
                                             sensor_img=True)
    end_time = time.time()
    print('touch', touch)
    print('total time: {}'.format(end_time - start_time))
    plt.show()
