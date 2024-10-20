"""
多元传感器融合
"""
import numpy as np
import math
import base64
import matplotlib.pyplot as plt
from io import BytesIO


def gen_multi_sensor_fusion_img():
    fig = plt.figure()
    plt.rcParams['axes.facecolor'] = '#343541'
    plt.rcParams['font.size'] = 10  # 字体大小
    plt.rcParams['xtick.color']='white'
    plt.rcParams['ytick.color']='white'
    # print(plt.rcParams)
    fig.patch.set_facecolor('#343541')
    fig.patch.set_alpha(0.9)
    # ax = plt.subplot(projection='polar')
    # ax=plt.figure()
    ax = plt.subplot()
    plt.imshow(np.random.rand(64, 64))
    # plt.show()
    width = np.pi / 4 * np.random.rand()
    # plt.show()
    save_file = BytesIO()
    plt.savefig(save_file, format="png")
    save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
    plt.clf()
    return save_file_base64

if __name__ == '__main__':
    gen_multi_sensor_fusion_img()