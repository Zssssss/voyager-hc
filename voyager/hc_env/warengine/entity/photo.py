import base64
from io import BytesIO

from sympy import *
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
import math
from mpl_toolkits.mplot3d import Axes3D
from geographiclib.geodesic import Geodesic
import matplotlib.image as mpimg # mpimg 用于读取图片

from SQ.util.utils import CoordinateTransformer
geod = Geodesic.WGS84

def gen_photo_img(touch=False):
    # if not touch:
    #     data = np.zeros((64, 64))
    # else:
    #     data = np.zeros((64, 64))
    #     data[30:50, 10:60] = np.random.rand(20, 50)
    #
    fig = plt.figure()
    plt.rcParams['axes.facecolor'] = '#343541'
    plt.rcParams['font.size'] = 10 # 字体大小
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    fig.patch.set_facecolor('#343541')
    fig.patch.set_alpha(0.9)
    # ax = plt.subplot()
    # plt.imshow(data, cmap='jet')
    #
    # # plt.show()

    lena = mpimg.imread('../model/photo.png')  # 读取和代码处于同一目录下的 lena.png
    plt.imshow(lena)
    save_file = BytesIO()
    plt.savefig(save_file, format="png")
    save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
    # plt.clf()
    # plt.close()
    return save_file_base64

if __name__ == '__main__':
    gen_photo_img(touch=True)
    plt.show()