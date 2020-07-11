import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

import astar
import reeds_shepp_path as rs
import lib


class C:    # Parameter config
    PI = np.pi
    VR = 1.0
    XY_RESO = 2.0  # [m]
    YAW_RESO = np.deg2rad(15.0)  # [rad]
    GOAL_TYAW_TH = np.deg2rad(5.0)  # [rad]
    Motion_RESO = 0.1  # [m] path interporate resolution
    N_STEER = 20.0  # number of steer command
    EXTEND_LEN = 5.0  # [m] map extend length
    SKIP_COLLISION_CHECK = 20  # skip number for collision check

    SWITCH_BACK_COST = 100.0  # switch back penalty cost
    BACKWARD_COST = 5.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost
    JACKKNIF_COST = 200.0  # Jackknif cost
    H_COST = 5.0  # Heuristic cost

    WB = lib.WB  # [m] Wheel base
    MAX_STEER = lib.MAX_STEER  # [rad] maximum steering angle


class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost


