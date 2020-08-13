"""
Hierarchical Optimization-Based Collision Avoidance
@author: Huiming Zhou
@paper: Autonomous Parking using Optimization-Based Collision Avoidance
@link: https://ieeexplore.ieee.org/document/8619433
@Julia version of HOBCA algorithm from author: https://github.com/XiaojingGeorgeZhang/H-OBCA
"""

import os
import sys
import math
import heapq
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import HybridAstarPlanner.hybrid_astar as HybridAStar
import HybridAstarPlanner.draw as draw


def pi_2_pi(theta):
    if theta > math.pi:
        return theta - 2.0 * math.pi

    if theta < -math.pi:
        return theta + 2.0 * math.pi

    return theta


def main():
    x, y = 51, 31
    sx, sy, syaw0 = 10.0, 7.0, np.deg2rad(120.0)
    gx, gy, gyaw0 = 45.0, 20.0, np.deg2rad(90.0)

    ox, oy = HybridAStar.design_obstacles(x, y)
    path = HybridAStar.hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,
                                             ox, oy, HybridAStar.C.XY_RESO, HybridAStar.C.YAW_RESO)

    x = path.x
    y = path.y
    yaw = path.yaw
    direction = path.direction

    for k in range(len(x)):
        plt.cla()
        plt.plot(ox, oy, "sk")
        plt.plot(x, y, linewidth=1.5, color='r')

        if k < len(x) - 2:
            dy = (yaw[k + 1] - yaw[k]) / HybridAStar.C.MOVE_STEP
            steer = pi_2_pi(math.atan(-HybridAStar.C.WB * dy / direction[k]))
        else:
            steer = 0.0

        draw.draw_car(gx, gy, gyaw0, 0.0, HybridAStar.C, 'dimgray')
        draw.draw_car(x[k], y[k], yaw[k], steer, HybridAStar.C)
        plt.title("Hybrid A*")
        plt.axis("equal")
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main()
