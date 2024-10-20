"""
1、飞行高度和速度有关系
2、转弯半径和速度有关系
"""
import numpy as np
import math

import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from simple_pid import PID
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84
import matplotlib.pyplot as plt

# 根据速度计算转弯半径
def cal_turn_course_by_vel(vel):
    r = (vel - 280) / 20 * 0.34 + 2.3
    t = 2 * r * 1000 / (vel * 1000 / 3600)
    course = 360 / t
    return course

def cal_subangle_dir(a0, a1):
    a0, a1 = a0 % 360, a1 % 360
    delta_angle, dir = 0, 1
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


class PlaneControl:
    def __init__(self, lat, lon, course, vel, alt, max_alt=8200, max_vel=400):
        self.lat = lat
        self.lon = lon
        self.course = course
        self.vel = vel
        self.alt = alt
        self.max_alt = max_alt
        self.min_alt = 50
        self.max_vel = max_vel


    def control(self, course, vel, height):
        alt_pid = self.alt_control(height)
        self.alt += alt_pid
        vel_pid = self.v_control(vel)
        self.vel += vel_pid
        course_pid = self.course_control(course)
        self.course += course_pid
        self.course %= 360
        if self.alt > self.max_alt:
            self.alt = self.max_alt
        elif self.alt < self.min_alt:
            self.alt = self.min_alt
        if self.vel > self.max_vel:
            self.vel = self.max_vel
        return self.course, self.vel, self.alt

    def alt_control(self, target_alt):
        kp = 0.01
        ki = 0.0005
        kd = 0.00001
        controler = PID(kp, ki, kd)
        error = target_alt - self.alt
        output = controler(-error)
        return output

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
        # error = target_course - self.course
        delta_angle, dir, origin_delta_angle = cal_subangle_dir(target_course, self.course)
        error = dir * delta_angle
        output = controler(error)
        return output

class PlaneAssist:
    def __init__(self, id, lat, lon, alt):
        self.lat = lat
        self.id = id
        self.lon = lon
        self.alt = alt
        self.routes = []
        self.route_id = 0

    def get_patrol_routes(self):
        azi0 = 30
        azi1 = -30
        azi2 =  150
        azi3 = -150
        s0 = 6000
        s1 = 15_000
        g = {
            "lat2": self.lat,
            "lon2": self.lon
        }
        g0 = geod.Direct(g["lat2"], g["lon2"], s12=s0, azi1=azi0)
        g1 = geod.Direct(g['lat2'], g['lon2'], s12=s1, azi1=azi0)
        g2 = geod.Direct(g['lat2'], g['lon2'], s12=s1, azi1=azi1)
        g3 = geod.Direct(g['lat2'], g['lon2'], s12=s0, azi1=azi1)
        g4 = geod.Direct(g['lat2'], g['lon2'], s12=s0, azi1=azi2)
        g5 = geod.Direct(g['lat2'], g['lon2'], s12=s1, azi1=azi2)
        g6 = geod.Direct(g['lat2'], g['lon2'], s12=s1, azi1=azi3)
        g7 = geod.Direct(g['lat2'], g['lon2'], s12=s0, azi1=azi3)

        self.routes.append(g)
        self.routes.append(g0)
        self.routes.append(g1)
        self.routes.append(g2)
        self.routes.append(g3)
        self.routes.append(g)
        self.routes.append(g4)
        self.routes.append(g5)
        self.routes.append(g6)
        self.routes.append(g7)

if __name__ == '__main__':
    plane = PlaneControl(lat=12,lon=110, course=0, vel=200, alt=200)
    agent = PlaneAssist(id=0, lat=plane.lat, lon=plane.lon, alt=plane.alt)
    rt_pos = {
        "lat": 12,
        "lon": 110
    }
    courses = []
    routes = []
    alts = []
    for angle in np.arange(0, 360, 10):
        g = geod.Direct(lat1=12, lon1=110, azi1=angle, s12=8_000)
        routes.append(g)
    route_id = 0
    agent.get_patrol_routes()
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
            course, vel, alt = plane.control(course=g['azi1'], height=500, vel=400)
            g = geod.Direct(plane.lat, plane.lon, s12=vel*1000/3600, azi1=course)
            lats.append(plane.lat)
            lons.append(plane.lon)
            plane.lat = g['lat2']
            plane.lon = g['lon2']
            alts.append(alt)
            vels.append(vel)
            courses.append(course)
        if i % 20 == 0:
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
            plt.subplot(224)
            plt.title('alts')
            plt.plot(alts)
            plt.show()
            plt.pause(0.001)
            plt.ioff()



    # for i in range(3600):
    #     target_route_id = agent.route_id % len(agent.routes)
    #     target_point = agent.routes[target_route_id]
    #     g = geod.Inverse(plane.lat, plane.lon, target_point['lat2'], target_point['lon2'])
    #     if g['s12'] < 600:
    #         agent.route_id += 1
    #     else:
    #         course, vel, alt = plane.control(course=g['azi1'], height=200, vel=200)
    #         g = geod.Direct(plane.lat, plane.lon, s12=vel * 1000 / 3600, azi1=course)
    #         lats.append(plane.lat)
    #         lons.append(plane.lon)
    #         courses.append(course)
    #         plane.lat = g['lat2']
    #         plane.lon = g['lon2']
    #     if i % 20 == 0:
    #         plt.ion()
    #         # plt.scatter(lats, lons)
    #         plt.subplot(121)
    #         plt.plot(courses)
    #         plt.subplot(122)
    #         plt.scatter(lats, lons)
    #         plt.show()
    #         plt.pause(0.001)
    #         plt.ioff()


                       # commands.append(
            #     PlaneCommand.move_control(obj_type=RedObjType.PLANE, id=id, velocity=200, height=500,
            #                               course=g['azi1']))