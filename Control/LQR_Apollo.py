"""
LQR controller for autonomous vehicle
@author: huiming zhou (zhou.hm0420@gmail.com)

This controller is the python version of LQR controller of Apollo.
GitHub link of BaiDu Apollo: https://github.com/ApolloAuto/apollo

In this file, we will use hybrid A* planner for path planning, while using
LQR with parameters in Apollo as controller for path tracking.
"""

import os
import sys
import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import HybridAstarPlanner.hybrid_astar as HybridAStar
import HybridAstarPlanner.draw as draw
from Control.lateral_controller_conf import *


class VehicleState:
    def __init__(self, x, y, yaw, v, gear):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.gear = gear


class Gear(Enum):
    FORWARD = 1
    BACKWARD = 2


class LatController:
    def __init__(self):
        self.x = 0


def main():
    x = Gear.BACKWARD
    if x == Gear.FORWARD:
        print(2)
    else:
        print(1)


if __name__ == '__main__':
    main()










