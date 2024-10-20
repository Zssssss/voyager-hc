"""
基于深度强化学习的无人潜航器智能对抗决策
声呐信号建模
以1000m距离为标准 速度6节
速度会印象峰值
距离会印象波宽和峰值
声呐方程
"""
import base64
from io import BytesIO

import numpy as np
import math
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84

"""
声呐模型
被动声呐浮标 
主动声呐浮标
海洋噪声
# """


# class SonarModel:
#     def __init__(self):
#         pass
#
#     def
#
#     def active_sonar(self):
#         pass
#
#     def passive_sonar(self):
#         pass
def normality(x, deg):
    mean = deg
    std = 1 / (0.1)
    A = 10 * (np.sqrt(2 * math.pi) * std)
    return A / (np.sqrt(2 * math.pi) * std) * np.exp(-(x - mean) ** 2 / (2 * std * std))


def get_red_sonar_img(plane_lat, plane_lon, sub_lat, sub_lon, sonar_type, sensor_img=False):
    course = None
    lat = None
    lon = None
    touch = 0
    g = geod.Inverse(plane_lat, plane_lon, sub_lat, sub_lon)
    dis = g["s12"]
    # print('dis', dis)
    save_file_base64 = None

    # 拖拽声呐，探测范围是圆环
    if (dis >= 0.5 * 1852 and dis <= 1 * 1852) or (dis >= 1.5 * 1852 and dis <= 2 * 1852) or (
            dis >= 2.5 * 1852 and dis <= 3 * 1852) or (dis >= 3.5 * 1852 and dis <= 4 * 1852) or (
            dis >= 4.5 * 1852 and dis <= 5 * 1852) or (dis >= 5.5 * 1852 and dis <= 6 * 1852) or (
            dis >= 6.5 * 1852 and dis <= 7 * 1852) or (dis >= 7.5 * 1852 and dis <= 8 * 1852):
        touch = 1
        course = g['azi1']
        if sonar_type:
            point_num = 720
            angles = np.linspace(0, 360, point_num)
            r = normality(angles, deg=g["azi1"] % 360) + [np.random.random() for _ in range(point_num)]
            r /= max(r)
            lat = sub_lat
            lon = sub_lon
            if sensor_img:
                thetas = np.linspace(0, np.pi * 2, point_num)
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
                # plt.title('sonar')
                # plt.show()

                save_file = BytesIO()
                plt.savefig(save_file, format="png")
                save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
    else:
        touch = 0

    return save_file_base64, touch, course, lat, lon


def get_blue_sonar_img(blue_obs, red_obs, env_obs, sonar_type, error):
    course = []
    lat = []
    lon = []
    touch = []
    # 探测红方的声呐浮标
    for id in range(red_obs.uav_field.uav_nums):
        uav = red_obs.uav_field.uavs[id]
        for buoy in uav.buoys:
            if not buoy.dead:
                g = geod.Inverse(blue_obs.lat, blue_obs.lon, buoy.lat, buoy.lon)
                dis = g["s12"]
                g_virtual = geod.Direct(lat1=buoy.lat,
                                        lon1=buoy.lon,
                                        s12=np.random.random() * 100,
                                        azi1=np.random.random() * 360)
                g_ = geod.Inverse(blue_obs.lat, blue_obs.lon, g_virtual['lat2'], g_virtual['lon2'])
                if dis < 10000:
                    if buoy.btype == '68':
                        if sonar_type:
                            lat.append(g_virtual['lat2'])
                            lon.append(g_virtual['lon2'])
                            course.append(g_['azi1'])
                            touch.append(1)
                        else:
                            course.append(g_['azi1'])
                            lat.append(None)
                            lon.append(None)
                            touch.append(1)
    # 探测红方的拖曳声呐
    # for id in range(red_obs.usv_field.usv_nums):
    #     if red_obs.usv_field.usvs[id].sonar.sonar_type:
    #         g = geod.Inverse(blue_obs.lat, blue_obs.lon, red_obs.usv_field.usvs[id].lat,
    #                          red_obs.usv_field.usvs[id].lon)
    #         dis = g["s12"]
    #         if dis < 10000:
    #             if sonar_type:
    #                 lat.append(red_obs.usv_field.usvs[id].lat + error[0])
    #                 lon.append(red_obs.usv_field.usvs[id].lon + error[1])
    #                 course.append(g['azi1'] + error[2])
    #                 touch.append(1)
    #             else:
    #                 course.append(g['azi1'] + error[2])
    #                 lat.append(None)
    #                 lon.append(None)
    #                 touch.append(1)

    # 探测海洋环境信息（渔船等）
    # 探测渔船信息
    # for id in range(env_obs.fishing_boats):
    #     g = geod.Inverse(blue_obs.lat, blue_obs.lon, env_obs.fishing_boats[id].lat,
    #                      env_obs.fishing_boats[id].lon)
    #     dis = g["s12"]
    #     if dis < 10000:
    #         if sonar_type:
    #             lat.append(red_obs.usv_field.usvs[id].lat + error[0])
    #             lon.append(red_obs.usv_field.usvs[id].lon + error[1])
    #             course.append(g['azi1'] + error[2])
    #             touch.append(1)
    #         else:
    #             course.append(g['azi1'] + error[2])
    #             lat.append(None)
    #             lon.append(None)
    #             touch.append(1)
    return lat, lon, course, touch


def get_buoy_img(plane_lat, plane_lon, sub_lat, sub_lon, buoy_channel, sensor_img=False):
    course = None
    lat = None
    lon = None
    touch = 0
    g = geod.Inverse(plane_lat, plane_lon, sub_lat, sub_lon)
    dis = g["s12"]
    # print('dis', dis)
    save_file_base64 = None
    if buoy_channel == 70:
        # 被动声呐浮标
        if dis <= 1000:
            touch = 1
            course = g['azi1']
            lat = sub_lat
            lon = sub_lon
    elif buoy_channel == 62:
        if dis <= 1200:
            touch = 1
            course = g['azi1']
    elif buoy_channel == 68:
        if dis <= 3000:
            touch = 1
            course = g['azi1']
    if touch and sensor_img:
        point_num = 720
        angles = np.linspace(0, 360, point_num)
        r = normality(angles, deg=g["azi1"] % 360) + [np.random.random() for _ in range(point_num)]
        r /= max(r)
        thetas = np.linspace(0, np.pi * 2, point_num)
        fig = plt.figure()
        plt.rcParams['axes.facecolor'] = '#343541'
        plt.rcParams['font.size'] = 10  # 字体大小
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        fig.patch.set_facecolor('#343541')
        fig.patch.set_alpha(0.9)
        ax = plt.subplot(projection='polar')
        # width = np.pi /  * np.random.rand()
        width = np.pi * 2 / point_num * 8
        colors = plt.get_cmap('autumn')(r)
        # plt.ylim((0, 0.1))
        ax.bar(thetas, r, width=width, color=colors, alpha=0.5)
        # plt.savefig('.\\buoy.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
        # plt.title('sonar')
        # plt.show()

        save_file = BytesIO()
        plt.savefig(save_file, format="png")
        save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')

    return save_file_base64, touch, course, lat, lon


# def get_blue_sonar_img(red_obs, sub_lat, sub_lon, sonar_type):
#     course = None
#     lat = None
#     lon = None
#     touch = 0
#     for obs in red_obs:


# g = geod.Inverse(plane_lat, plane_lon, sub_lat, sub_lon)
# dis = g["s12"]
# # print('dis', dis)
# save_file_base64 = None
# if agent_type == "usv":
#     # 拖拽声呐，探测范围是圆环
#     if (dis >= 0.5 * 1852 and dis <= 1 * 1852) or (dis >= 1.5 * 1852 and dis <= 2 * 1852) or (
#             dis >= 2.5 * 1852 and dis <= 3 * 1852) or (dis >= 3.5 * 1852 and dis <= 4 * 1852) or (
#             dis >= 4.5 * 1852 and dis <= 5 * 1852) or (dis >= 5.5 * 1852 and dis <= 6 * 1852) or (
#             dis >= 6.5 * 1852 and dis <= 7 * 1852) or (dis >= 7.5 * 1852 and dis <= 8 * 1852):
#         touch = 1
#         course = g['azi1']
#         if sonar_type:
#             point_num = 720
#             angles = np.linspace(0, 360, point_num)
#             r = normality(angles, deg=g["azi1"] % 360) + [np.random.random() for _ in range(point_num)]
#             r /= max(r)
#             lat = sub_lat
#             lon = sub_lon
#
#             thetas = np.linspace(0, np.pi * 2, point_num)
#             fig = plt.figure()
#             plt.rcParams['axes.facecolor'] = '#343541'
#             plt.rcParams['font.size'] = 10  # 字体大小
#             plt.rcParams['xtick.color'] = 'white'
#             plt.rcParams['ytick.color'] = 'white'
#             # print(plt.rcParams)
#             fig.patch.set_facecolor('#343541')
#             fig.patch.set_alpha(0.9)
#             ax = plt.subplot(projection='polar')
#             # width = np.pi /  * np.random.rand()
#             width = np.pi * 2 / point_num * 8
#             colors = plt.get_cmap('autumn')(r)
#             # plt.ylim((0, 0.1))
#             ax.bar(thetas, r, width=width, color=colors, alpha=0.5)
#             # plt.show()
#
#             save_file = BytesIO()
#             plt.savefig(save_file, format="png")
#             save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
#     else:
#         touch = 0


# def get_buoy_img(plane_lat, plane_lon, sub_lat, sub_lon):
#     g = geod.Inverse(plane_lat, plane_lon, sub_lat, sub_lon)
#     dis = g["s12"]
#     touch = False
#     course = None
#     point_num = 720
#     if touch:
#         angles = np.linspace(0, 360, point_num)
#         r = normality(angles, deg=deg, v=v, dis=dis) + [np.random.random() for _ in range(point_num)]
#         r /= max(r)
#         thetas = np.linspace(0, np.pi * 2, point_num)
#     else:
#         thetas = np.linspace(0, np.pi * 2, point_num)
#         r = [np.random.random() * 0.1 for _ in range(point_num)]
#     thetas = np.linspace(0, np.pi * 2, point_num)
#
#     fig = plt.figure()
#     plt.rcParams['axes.facecolor'] = '#343541'
#     plt.rcParams['font.size'] = 10  # 字体大小
#     plt.rcParams['xtick.color'] = 'white'
#     plt.rcParams['ytick.color'] = 'white'
#     # print(plt.rcParams)
#     fig.patch.set_facecolor('#343541')
#     fig.patch.set_alpha(0.9)
#     ax = plt.subplot(projection='polar')
#     # width = np.pi /  * np.random.rand()
#     width = np.pi * 2 / point_num * 8
#     colors = plt.get_cmap('autumn')(r)
#     # plt.ylim((0, 0.1))
#     ax.bar(thetas, r, width=width, color=colors, alpha=0.5)
#     plt.show()
#     save_file = BytesIO()
#     plt.savefig(save_file, format="png")
#     save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
#     # plt.clf()
#     # plt.close()
#     return save_file_base64, touch, course


# 生成正态分布信号
# def normality(x, deg, v, dis):
#     mean = deg
#     std = 1000 / (dis + 1000) * 30
#     # altitude = v / np.exp(dis - 1000) * 3 if v >= 6 else 1
#     altitude = v / 6 * 3 if v >= 6 else 1
#     A = altitude * (np.sqrt(2 * math.pi) * std)
#     return A / (np.sqrt(2 * math.pi) * std) * np.exp(-(x - mean) ** 2 / (2 * std * std))


if __name__ == '__main__':
    import time

    start_time = time.time()
    # img, touch, course = get_buoy_img(plane_lat=11.9, plane_lon=120, sub_lat=12.001, sub_lon=120)
    # end_time = time.time()
    # print('total time: {}'.format(end_time - start_time))
    # plt.show()
    # print(get_red_sonar_img(plane_lat=12, plane_lon=120.1, sub_lat=12.001, sub_lon=120, sonar_type=1, sensor_img=True))
    get_buoy_img(plane_lat=14, plane_lon=120.1, sub_lat=12.001, sub_lon=120, buoy_channel=70, sensor_img=True)
