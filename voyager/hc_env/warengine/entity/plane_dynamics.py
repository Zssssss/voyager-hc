"""
飞机动力学建模

last_course： 上一个时刻的航向
last_vel： 上一个时刻的速度
last_height： 上一个时刻的高度

cur_course, cur_vel, cur_height = move_control(target_course, target_vel, target_height)
"""
"""
飞机动力学建模 
last_course: 上一个时刻的航向
last_vel: 上一个时刻的速度
last_height: 上一个时刻的高度 
"""
# 可视化三维轨迹图
import matplotlib.pyplot as plt
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def control(x_input, control):
    g = 9.8 # 重力加速度
    velocity, gamma, fai = x_input # 机体速度、俯仰角、航向角
    nx, nz, roll = control # 切向推力 法向推力 横滚角


"""
飞行器控制类
"""
class PlaneControl:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0 # 高度
        self.v = 0 # 机体速度
        self.gamma = 0
        self.fai = 0
        self.course = 0 # 航向

    def update(self, target_course, target_vel, target_height):
        pass

def plane_move_control(last_course, last_vel, last_height,
                       target_course, target_vel, target_height):
    cur_course = 0
    cur_vel = 0
    cur_height = 0
    return cur_course, cur_vel, cur_height

if __name__ == '__main__':
    x, y, z = 0, 0, 0
    v, gamma, fai = 100 * 1000 / 3600, 0, math.pi / 2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    xx = []
    yy = []
    zz = []
    T = 165
    g = 9.8
    dt = 1
    xx.append(x)
    yy.append(y)
    zz.append(z)
    for t in range(T):
        if t <= 40: # 直线
            nx, nz, roll = np.sin(gamma), np.cos(gamma), 0
        elif t <= 60: #左转
            nx = np.sin(gamma)
            roll = -np.pi / 10
            nz = np.cos(gamma) / np.cos(roll)
        elif t <= 80: # 直线
            nx, nz, roll = np.sin(gamma), np.cos(gamma), 0
        elif t <= 100: #右转
            nx = np.sin(gamma)
            roll = np.pi / 10
            nz = np.cos(gamma) / np.cos(roll)
        elif t <= 120: #上升
            nx = np.sin(gamma)
            roll = 0
            nz = np.cos(gamma) + 0.3
        elif t <= 130: # 直线
            nx, nz, roll = np.sin(gamma), np.cos(gamma), 0
        elif t <= 160: # 下降
            nx = np.sin(gamma)
            roll = 0
            nz = np.cos(gamma) - 0.3
        dv = g * (nx - np.sin(gamma))
        dgamma = g * (nz * np.cos(roll) - np.cos(gamma)) / v
        dfai = g * nz * np.sin(roll) / (v * np.cos(gamma))
        v = v + dv
        gamma = gamma + dgamma
        fai = fai + dfai
        dx = v * np.cos(gamma) * np.sin(fai)
        dy = v * np.cos(gamma) * np.cos(fai)
        dz = v * np.sin(gamma)
        x = x + dx
        y = y + dy
        z = z + dz
        xx.append(x)
        yy.append(y)
        zz.append(z)
        print(t, v, gamma, fai)
    ax.plot(xx, yy, zz)
    plt.title('plane dynamics trajectory')
    plt.show()