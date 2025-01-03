import numpy as np
import math

import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from simple_pid import PID
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84
import matplotlib.pyplot as plt
def cal_subangle_dir(a0, a1):
    a0, a1 = a0 % 360, a1 % 360
    origin_delta_angle = abs(a0 - a1)
    if abs(a0 - a1) <= 180:
        delta_angle = abs(a0 - a1)
        if a1 >= a0:
            dir = 1
        else:
            dir = -1
    else:
        delta_angle = 360 - abs(a0 - a1)
        if 0 <= a1 <= 180:
            dir = 1
        else:
            dir = -1
    return delta_angle, dir, origin_delta_angle


class UsvControl:
    def __init__(self, lat, lon, course, vel):
        self.lat = lat
        self.lon = lon
        self.course = course
        self.vel = vel
        self.max_vel = 60


    def control(self, course, vel):
        vel_pid = self.v_control(vel)
        self.vel += vel_pid
        course_pid = self.course_control(course)
        self.course += course_pid
        self.course %= 360
        if self.vel > self.max_vel:
            self.vel = self.max_vel
        return self.course, self.vel


    def v_control(self, target_vel):
        kp = 0.01
        ki = 0.005
        kd = 0.00001
        controler = PID(kp, ki, kd)
        error = target_vel - self.vel
        output = controler(-error)
        return output

    def course_control(self, target_course):
        kp = 0.1
        ki = 0.005
        kd = 0.00001
        controler = PID(kp, ki, kd)
        delta_angle, dir, origin_delta_angle = cal_subangle_dir(target_course, self.course)
        error = dir * delta_angle
        output = controler(error)
        return output


if __name__ == '__main__':
    plane = UsvControl(lat=12,lon=110, course=0, vel=10)
    rt_pos = {
        "lat": 12,
        "lon": 110
    }
    courses = []
    routes = []
    for angle in np.arange(0, 360, 10):
        g = geod.Direct(lat1=12, lon1=110, azi1=angle, s12=1_000)
        routes.append(g)
    route_id = 0
    # agent.get_patrol_routes()
    lats = []
    lons = []
    vels = []
    for i in range(3600):
        target_id = route_id % len(routes)
        target_point = routes[target_id]
        g = geod.Inverse(plane.lat, plane.lon, target_point['lat2'], target_point['lon2'])
        # print(g['s12'])
        if g['s12'] < 800:
            route_id += 1
        else:
            course, vel = plane.control(course=g['azi1'], vel=12)
            print(g['s12'],course, vel)
            g = geod.Direct(plane.lat, plane.lon, s12=vel*1852/3600, azi1=course)
            lats.append(plane.lat)
            lons.append(plane.lon)
            plane.lat = g['lat2']
            plane.lon = g['lon2']
            vels.append(vel)
            courses.append(course)
        if i % 10 == 0:
            plt.ion()
            # plt.scatter(lats, lons)
            plt.subplot(221)
            plt.title('courses')
            plt.plot(courses)
            plt.subplot(222)
            plt.title('pos')
            plt.scatter(lats, lons)
            plt.subplot(223)
            plt.title('vels')
            plt.plot(vels)
            plt.show()
            plt.pause(0.001)
            plt.ioff()

