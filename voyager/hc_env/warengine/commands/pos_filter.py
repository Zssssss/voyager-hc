
import numpy as np
import pandas as pd
from pyproj import Transformer
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84

import random
import matplotlib.pyplot as plt

def filter_v0(adc_list):
    lats = []
    lons = []
    idx0 = 30
    idx1 = 0
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
                    print(g['s12'])
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
            mean_lat = np.mean(lats[-idx1:]) * 0.5 + 0.5 * mean_lat
        if lons:
            mean_lon = np.mean(lons[-idx1:]) * 0.5 + 0.5 * mean_lon

        lats.append(mean_lat)
        lons.append(mean_lon)
    return lats, lons


class Kalman:
    def __init__(self, F, B, H, Q, X, P):
        # 固定参数
        self.F = F  # 状态转移矩阵
        self.B = B  # 控制矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声
        # self.R_infrared = R_infrared  # 光电量测噪声
        # 迭代参数
        self.X_posterior = X  # 后验状态, 定义为 [中心x,中心y,宽w,高h,dx,dy]
        self.P_posterior = P  # 后验误差矩阵
        self.X_prior = None  # 先验状态
        self.P_prior = None  # 先验误差矩阵
        self.K1 = None  # kalman gain
        self.K2 = None  # kalman gain
        self.Z = None  # 观测

    def predict(self, X, P):
        """
        预测外推
        :return:
        """
        # self.X_prior = np.dot(self.F, self.X_posterior)
        # self.P_prior = np.dot(np.dot(self.F, self.P_posterior), self.F.T) + self.Q
        X = np.matmul(self.F, X)
        P = np.matmul(np.matmul(self.F, P), self.F.transpose()) + self.Q

        return X, P

    def update(self, X, P, R, mea=None):
        """
        完成一次kalman滤波

        """

        Z = mea
        if mea is not None:
           K = np.dot(np.dot(P, self.H.T),
                           np.linalg.inv(np.dot(np.dot(self.H, P),
                                                self.H.T) + R))  # 计算卡尔曼增益
           X = X + np.dot(K, Z - np.dot(self.H, X))  # 更新后验估计
           P = np.dot(np.eye(self.H.shape[0]) - np.dot(K, self.H),
                       P)  # 更新后验误差矩阵

        return X, P

# lat = 30
# lon = 120
# adc_list = []
# for i in range(10):
#     adc_list.append([lat, lon])
#     lat += np.random.rand() * 0.001
#     lon += np.random.rand() * 0.001
# filter_lats, filter_lons = filter_v0(adc_list)
# a = np.array(adc_list)[:, 0]
# b = np.array(adc_list)[:, 1]
# plt.plot(a, b)
# plt.plot(filter_lats, filter_lons, 'r.')
# plt.show()


# # 状态转移矩阵，上一时刻的状态转移到当前时刻
# F = np.array([[1, 0, 1, 0],
#               [0, 1, 0, 1],
#               [0, 0, 1, 0],
#               [0, 0, 0, 1]])
# # 控制输入矩阵B
# B = None
# # 过程噪声协方差矩阵Q，p(w)~N(0,Q)，噪声来自真实世界中的不确定性、智能体移动的不确定性（突然加速、减速、转弯等）
# Q = np.eye(F.shape[0]) * 0.1
# # 状态观测矩阵
# H = np.array([[1, 0, 0, 0],
#               [0, 1, 0, 0],
#               [0, 0, 0, 0],
#               [0, 0, 0, 0]])
# # 状态估计协方差矩阵P初始化
# P = np.eye(F.shape[0])
# # 读取数据
# lat = 30
# lon = 120
# adc_list = []
# adc_list_k = []
#
# X = np.array([lat, lon, 0, 0]).reshape(-1,1)
#
# # X = np.zeros((F.shape[1], 1))
# k = Kalman(F, B, H, Q, X, P)
#
# for i in range(10):
#     measure_infrared = np.array([lat, lon, 0, 0]).reshape(-1, 1)
#     X, P = k.predict(X, P)
#     X, P = k.update(X, P, R=np.eye(F.shape[0]) * 0.1, mea=measure_infrared)  # 光电传感器更新
#     adc_list_k.append([X[0], X[1]])
#
#     adc_list.append([lat, lon])
#     lat += np.random.rand() * 0.01
#     lon += np.random.rand() * 0.01
#
# plt.plot(np.array(adc_list)[:, 0], np.array(adc_list)[:, 1])
# plt.plot(np.array(adc_list_k)[:, 0], np.array(adc_list_k)[:, 1], 'r.')
# plt.show()


import numpy as np
import math
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt

geod = Geodesic.WGS84
import time


# import Geodesic.WGS84 as geod
# create kalman filter data
def create_data():
    direction = np.random.uniform(0, 360)
    lat, lon = 17, 110
    pos_list = [[lat, lon]]
    dt = 1
    for t in range(200):
        if t < 2000:
            v = np.random.uniform(11, 13)
        elif t < 5000:
            v = np.random.uniform(4,6)
        else:
            v = np.random.uniform(8,10)
        if t % 900 < 3 * 60:
            sub_dir = direction + 35
        else:
            sub_dir = direction - 15
        s = v * 1.83 * 1000 / 3600 * dt
        g = geod.Direct(lat1=lat, lon1=lon, s12=s, azi1=sub_dir)
        lat, lon = g["lat2"], g["lon2"]
        pos_list.append([lat, lon])

    adc_list = []
    for i, pos in enumerate(pos_list):
        lat, lon = pos
        g = geod.Direct(lat1=lat, lon1=lon, s12=np.random.random() * 500, azi1=np.random.random() * 360)
        adc_list.append([g["lat2"], g["lon2"]])

    return pos_list, adc_list


# without kalman filter
def filter_v0(adc_list):
    lats = []
    lons = []
    idx0 = 30
    idx1 = 20
    print(len(adc_list))
    history_dict = {}
    for i, adc_pos in enumerate(adc_list):
        d = []
        for j in range(max(0, i - idx0), i + 1):
            for k in range(j, i + 1):
                key = str(j) + ',' + str(k)
                if  key in history_dict.keys():
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

# kalman filter
def filter_v1(adc_list):
    lats, lons = [], []
    return lats, lons


def kalman_filter(adc_list):
    # 得到滤波之后的点
    pass


if __name__ == '__main__':
    import time
    pos_list, adc_list = create_data()
    lats = list(map(lambda x: x[0], pos_list))
    lons = list(map(lambda x: x[1], pos_list))

    adc_lats = list(map(lambda x: x[0], adc_list))
    adc_lons = list(map(lambda x: x[1], adc_list))

    # move average
    start_time = time.time()
    filter_lats, filter_lons = filter_v0(adc_list)
    end_time = time.time()
    # kalman filter
    kalman_filter(adc_list)
    print('total time: {}'.format(end_time - start_time))
    plt.title('track of submarine')
    # plt.scatter(adc_lats, adc_lons, marker='o', s=5, c='blue', label='adc pos')
    plt.scatter(lats, lons, marker='o', s=5, c='red', label='true pos')
    plt.scatter(filter_lats, filter_lons, marker='o', s=10, c='green', label='filter pos')
    plt.legend()
    plt.show()
