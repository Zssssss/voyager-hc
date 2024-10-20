"""
磁探建模
"""
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import random
from geographiclib.geodesic import Geodesic

geod = Geodesic.WGS84
from pyproj import CRS

crs_WGS84 = CRS.from_epsg(4326)
crs_WebMercator = CRS.from_epsg(3857)  # Web墨卡托投影坐标系
from pyproj import Transformer
from pyproj import Transformer
import base64
from io import BytesIO


def lla_to_xyz(lat, lon, alt):
    transprojr = Transformer.from_crs(
        "EPSG:4326",
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True)
    x, y, z = transprojr.transform(lon, lat, alt, radians=False)
    return x, y, z


def gauss_noisy(x, y, mu=0, sigma=0.05):
    """
    对输入数据加入高斯噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    x += random.gauss(mu, sigma)
    y += random.gauss(mu, sigma)
    return x, y


def my_noisy(x, y, p=0.2, max_noise=5):
    """
    对输入数据加入噪声
    :param x: x轴数据
    :param y: y轴数据
    :return:
    """
    p_noise = np.random.random()
    if p_noise < p:
        noise = np.random.randint(-max_noise, max_noise)
        new_x = x + noise
        noise = np.random.randint(-max_noise, max_noise)
        new_y = y + noise
        return new_x, new_y
    return x, y


class MagneticModel:
    '''
    步骤1：无人机和目标坐标，gps2xyz
    步骤2：建模目标磁场
    步骤3：根据无人机位置，转换坐标原点，
    '''

    def __init__(self, detection_range=10000, sub_num=2):
        self.sub_num = sub_num
        self.detection_range = detection_range
        self.abs_M = 1 * 10 ** 6
        self.sub_info = []
        for i in range(0, sub_num):
            x_pro = random.random()
            y_pro = random.uniform(0, math.sqrt(1 - x_pro * x_pro))
            z_pro = math.sqrt(1 - x_pro * x_pro - y_pro * y_pro)
            # print(x_pro,y_pro,z_pro,x_pro * x_pro + y_pro * y_pro + z_pro * z_pro)
            Mx = self.abs_M * x_pro
            My = self.abs_M * y_pro
            Mz = self.abs_M * z_pro
            # print(self.Mx,self.My,self.Mx)
            self.sub_info.append({"Mx": Mx, "My": My, "Mz": Mz})

    def get_pos_log_magnetic(self, sub_id, x, y, z):
        # 得到坐标为（x,y,z）位置的磁场大小，取log
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0:
            return math.log10(self.abs_M)
        Mx = self.sub_info[sub_id]["Mx"]
        My = self.sub_info[sub_id]["My"]
        Mz = self.sub_info[sub_id]["Mz"]
        ax = 1 / (4 * math.pi) * (math.pow(x, 2) / math.pow(r, 5) - 1 / math.pow(r, 3))
        ay = 3 * x * y / (4 * math.pi * math.pow(r, 5))
        az = 3 * x * z / (4 * math.pi * math.pow(r, 5))
        bx = ay
        by = 1 / (4 * math.pi) * (3 * math.pow(y, 2) / math.pow(r, 5) - 1 / math.pow(r, 3))
        bz = 3 * y * z / (4 * math.pi * math.pow(r, 5))
        cx = az
        cy = bz
        cz = 1 / (4 * math.pi) * (3 * math.pow(z, 2) / math.pow(r, 5) - 1 / math.pow(r, 3))
        Hx = ax * Mx + ay * My + az * Mz
        Hy = bx * Mx + by * My + bz * Mz
        Hz = cx * Mx + cy * My + cz * Mz
        H = math.sqrt(math.pow(Hx, 2) + math.pow(Hy, 2) + math.pow(Hz, 2))
        return math.log10(H)

    def get_range_log_magnetic(self, sub_id, min_x=-100, max_x=100, min_y=-100, max_y=100, dis_z=1000):
        Distance_unit = 100  # 距离尺度
        # print(min_x, max_x, min_y, max_y)
        min_x = int(min_x / Distance_unit)
        max_x = int(max_x / Distance_unit)
        min_y = int(min_y / Distance_unit)
        max_y = int(max_y / Distance_unit)
        dis_z = int(dis_z / Distance_unit)
        # print(min_x, max_x, min_y, max_y)
        max_x = min_x - 1 + int(2 * self.detection_range / Distance_unit)
        max_y = min_y - 1 + int(2 * self.detection_range / Distance_unit)
        # print(min_x,max_x,min_y,max_y)
        values = np.zeros((max_y - min_y + 1, max_x - min_x + 1))
        # print(values.shape)
        for x in range(min_x, max_x + 1):
            # print(x-min_x)
            for y in range(min_y, max_y + 1):
                # new_x,new_y=my_noisy(x, y,p=0.2,max_noise=5)
                new_x, new_y = gauss_noisy(x, y, mu=0, sigma=1.5)
                H = self.get_pos_log_magnetic(sub_id, new_x, new_y, dis_z)
                values[y - min_y][x - min_x] = H  # +random.gauss(mu=0, sigma=0.1)
                # print(x - max_x,y - max_y,H)
        return values

    def cal_magnetic(self, sub_pos, plane_lat, plane_lon, plane_alt, dir='./', round=1, coding=1, sensor_img=False):
        plane_x, plane_y, _ = lla_to_xyz(plane_lat, plane_lon, plane_alt)
        plane_z = plane_alt
        detection_max_x = self.detection_range  # 以飞机为坐标原点
        detection_min_x = - self.detection_range
        detection_max_y = self.detection_range
        detection_min_y = -self.detection_range
        find_all = []
        self.sub_num = len(sub_pos)
        for i in range(0, self.sub_num):
            sub_lat = sub_pos[i]["sub_lat"]
            sub_lon = sub_pos[i]["sub_lon"]
            find = 0
            sub_z = sub_pos[i]["sub_alt"]
            sub_x, sub_y, _ = lla_to_xyz(sub_lat, sub_lon, sub_z)
            # print("(", plan_x, ",", plan_y, ")，", "(", sub_x, ",", sub_y, ")")
            dis_x = sub_x - plane_x
            dis_y = sub_y - plane_y
            dis_z = abs(sub_z - plane_z)
            dis = math.sqrt(dis_x * dis_x + dis_y * dis_y + dis_z * dis_z)
            # print(int(dis_x/100),int(dis_y/100),int(dis_z/100))
            value = self.get_range_log_magnetic(i,
                                                min_x=detection_min_x - dis_x,
                                                max_x=detection_max_x - dis_x,
                                                min_y=detection_min_y - dis_y,
                                                max_y=detection_max_y - dis_y,
                                                dis_z=dis_z)
            n = np.sum(value >= 1.5)
            if n > 1:
                find = 1
            find_all.append(find)
            value[value < 0] = 0
            if i == 0:
                res = value
            else:
                res = res + value

        # import os
        # if not os.path.exists(dir):
        #     os.makedirs(dir)

        # plt.savefig(dir + str(round) + '_' + '{}.png'.format(coding), bbox_inches='tight', pad_inches=0.0, dpi=300)

        # save_file = BytesIO()
        # plt.savefig(save_file, format="png")
        # save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
        # plt.show()
        if any(find_all) and sensor_img:
            plt.imshow(res, cmap='jet', vmax=8, vmin=0, origin='lower')
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            # plt.title('mag')
            # plt.show()
            save_file = BytesIO()
            plt.savefig(save_file, format="png")
            save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
        else:
            save_file_base64 = None
        return save_file_base64, find_all

    def draw_magnetic(self, plane_lat, plane_lon, plane_alt, sub_pos):
        a = 0.04
        b = 0.04
        for i in range(0, 10):
            plane_lat -= a / 10
            plane_lon += b / 10
            self.cal_magnetic(sub_pos, plane_lat, plane_lon, plane_alt)


if __name__ == '__main__':
    import time

    sub_num = 3
    sub_pos = []
    '''
    for i in range(0, sub_num):
        sub_lat = random.uniform(19, 19.12)
        sub_lon = random.uniform(110.4, 110.6)
        sub_alt = random.uniform(-35, -10)
        sub_pos.append({"sub_lat": sub_lat, "sub_lon": sub_lon, "sub_alt": sub_alt})
        print({"sub_lat": sub_lat, "sub_lon": sub_lon, "sub_alt": sub_alt})
    '''
    sub_pos = [{"sub_lat": 18.01944, "sub_lon": 112.5194, "sub_alt": -30}
               ]
    plane_lat = 18.01244
    plane_lon = 112.5294
    plane_alt = 1700
    start_time = time.time()
    magnetic = MagneticModel(detection_range=10000, sub_num=sub_num)
    img, find = magnetic.cal_magnetic(sub_pos, plane_lat, plane_lon, plane_alt)
    print(find)

    # magnetic.draw_magnetic(plane_lat, plane_lon, plane_alt, sub_pos)
    end_time = time.time()
    print('total time: {}'.format(end_time - start_time))
