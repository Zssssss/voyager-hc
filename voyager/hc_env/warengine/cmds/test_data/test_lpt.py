"""
# python
# -*- coding:utf-8 -*-
@Project : SQ_ChenWei
@File : test_lpt.py
@Author : 一杯可乐
@Time : 2024/4/1 9:57
@Description : 
"""
import matplotlib.pyplot as plt

from geographiclib.geodesic import Geodesic

from SQ.commands.plane_command import PlaneCommand, RedObjType
from SQ.entity.observation import BlueGlobalObservation, RedGlobalObservation

import numpy as np

geod = Geodesic.WGS84

a_lat = 17.5
a_lon = 114.3

b_lat = 16.5
b_lon = 114.6

screw_angle1 = geod.Inverse(a_lat, a_lon, b_lat, b_lon)["azi1"]
screw_angle2 = geod.Inverse(b_lat, b_lon, a_lat, a_lon)["azi1"]
print(screw_angle1)
print(screw_angle2)

plt.plot(a_lon, a_lat, 'go')
plt.plot(b_lon, b_lat, 'ro')
plt.show()
