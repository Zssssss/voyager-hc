from abc import ABC
import time
import numpy as np
from jsbsim_simulator import Simulation
from jsbsim_aircraft import Aircraft, x8
import jsbsim_properties as prp
from simple_pid import PID
from autopilot import X8Autopilot
from typing import Type, Tuple, Dict

import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
geod = Geodesic.WGS84


