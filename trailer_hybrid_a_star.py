import numpy as np
import math
import matplotlib.pyplot as plt
import heapq

import grid_a_star
import rs_path
import trailerlib

PI = np.pi

XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
GOAL_TYAW_TH = np.deg2rad(5.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interporate resolution
N_STEER = 20.0  # number of steer command
EXTEND_AREA = 5.0  # [m] map extend length
SKIP_COLLISION_CHECK = 4  # skip number for collision check

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle penalty cost
JACKKNIF_COST = 200.0  # Jackknif cost
H_COST = 5.0  # Heuristic cost

WB = trailerlib.WB  # [m] Wheel base
LT = trailerlib.LT  # [m] length of trailer
MAX_STEER = trailerlib.MAX_STEER  # [rad] maximum steering angle


class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, yaw1, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw1 = yaw1
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Config:
    def __init__(self, minx, miny, minyaw, minyawt, maxx, maxy,
                 maxyaw, maxyawt, xw, yw, yaww, yawtw, xyreso, yawreso):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.minyawt = minyawt
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.maxyawt = maxyawt
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.yawtw = yawtw
        self.xyreso = xyreso
        self.yawreso = yawreso


class Path:
    def __init__(self, x, y, yaw, yaw1, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw1 = yaw1
        self.direction = direction
        self.cost = cost


def calc_hybrid_astar_path(sx, sy, syaw, syaw1, gx, gy, gyaw, gyaw1, ox, oy, xyreso, yawreso):
    """
    sx: start x position [m]
    sy: start y position [m]
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xyreso: grid resolution [m]
    yawreso: yaw angle resolution [rad]
    """

    syaw, gyaw = rs_path.pi_2_pi(syaw), rs_path.pi_2_pi(gyaw)
    kdtree = set()
    ox, oy = ox[:], oy[:]

    for i in range(len(ox)):
        kdtree.add((ox[i], oy[i]))

    c = calc_config(ox, oy, xyreso, yawreso)
    nstart = Node(round(sx / xyreso), round(sy / xyreso), round(syaw / yawreso),
                  True, [sx], [sy], [syaw], [syaw1], [True], 0.0, 0.0, -1)
    ngoal = Node(round(gx / xyreso), round(gy / xyreso), round(gyaw / yawreso),
                 True, [gx], [gy], [gyaw], [gyaw1], [True], 0.0, 0.0, -1)

    h_dp = calc_holonomic_with_obstacle_heuristic(ngoal, ox, oy, xyreso)  # cost of each node

    openset, closed_set = dict(), dict()
    fnode = None
    openset[calc_index(nstart, c)] = nstart
    pq = []
    heapq.heappush(pq, (calc_cost(nstart, h_dp, ngoal, c), calc_index(nstart, c)))

    u, d = calc_motion_inputs()
    nmotion = len(u)

    while True:
        if not openset:
            print("Error: Cannot find path, No open set")
            return []

        _, c_id = heapq.heappop(pq)
        current = openset[c_id]

        openset.pop(c_id)
        closed_set[c_id] = current

        isupdated, fpath = update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree, gyaw1)


def update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree, gyaw1):
    apath = analystic_expantion(current, ngoal, c, ox, oy, kdtree)


def analystic_expantion(n, ngoal, c, ox, oy, kdtree):
    sx = n.x[-1]
    sy = n.y[-1]
    syaw = n.yaw[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs_path.calc_all_paths(sx, sy, syaw, ngoal.x[-1], ngoal.y[-1],
                                   ngoal.yaw[-1], max_curvature, step_size=MOTION_RESOLUTION)
    if len(paths) == 0:
        return None

    pathqueue = []
    for path in paths:
        steps = MOTION_RESOLUTION * path.directions
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yaw1[-1], steps)
        heapq.heappush(pathqueue, (calc_rs_path_cost(path, yaw1), path))

    for i in range(len(pathqueue)):
        _, path = heapq.heappop(pathqueue)

        steps = MOTION_RESOLUTION * path.directions
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yaw1[-1], steps)
        ind = range(0, len(path.x), SKIP_COLLISION_CHECK)

        if trailerlib.check_trailer_collision(ox, oy, path.x[ind], path.y[ind], path.yaw[ind], yaw1[ind], kdtree = kdtree)


def calc_rs_path_cost(rspath, yaw1):
    cost = 0.0
    for length in rspath.lengths:
        if length >= 0:
            cost += 1
        else:
            cost += abs(length) * BACK_COST

    for i in range(len(rspath.lengths)-1):
        if rspath.lengths[i]*rspath.lengths[i+1] < 0.0:
            cost += SB_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += STEER_COST*abs(MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes)-1):
        cost += STEER_CHANGE_COST*abs(ulist[i+1] - ulist[i])

    cost += JACKKNIF_COST * sum([abs(rs_path.pi_2_pi(x - y)) for x, y in zip(rspath.yaw, yaw1)])

    return cost


def calc_motion_inputs():
    up = [i for i in range(MAX_STEER / N_STEER, MAX_STEER, MAX_STEER / N_STEER)]
    u = [0.0] + [i for i in up] + [-i for i in up]
    d = [1.0 for _ in range(len(u))] + [-1.0 for _ in range(len(u))]
    u = u + u

    return u, d


def calc_cost(n, h_dp, ngoal, c):
    return n.cost + H_COST * h_dp[n.xind - c.minx, n.yind - c.miny]


def calc_index(node, c):
    ind = (node.yawind - c.minyaw) * c.xw * c.yw + (node.yind - c.miny) * c.xw + (node.xind - c.minx)

    # 4D grid
    yaw1ind = round(node.yaw1[-1] / c.yawreso)
    ind += (yaw1ind - c.minyawt) * c.xw * c.yw * c.yaww

    if ind <= 0:
        print("Error(calc_index):", ind)
    return ind


def calc_holonomic_with_obstacle_heuristic(gnode, ox, oy, xyreso):
    h_dp = grid_a_star.calc_dist_policy(gnode.x[-1], gnode.y[-1], ox, oy, xyreso, 1.0)
    return h_dp


def calc_config(ox, oy, xyreso, yawreso):
    min_x_m = min(ox) - EXTEND_AREA
    min_y_m = min(oy) - EXTEND_AREA
    max_x_m = max(ox) + EXTEND_AREA
    max_y_m = max(oy) + EXTEND_AREA

    ox.append(min_x_m)
    oy.append(min_y_m)
    ox.append(max_x_m)
    ox.append(max_y_m)

    minx = round(min_x_m / xyreso)
    miny = round(min_y_m / xyreso)
    maxx = round(max_x_m / xyreso)
    maxy = round(max_y_m / xyreso)

    xw = round(maxx - minx)
    yw = round(maxy - miny)

    minyaw = round(-PI / yawreso) - 1
    maxyaw = round(PI / yawreso)
    yaww = round(maxyaw - minyaw)

    minyawt = minyaw
    maxyawt = maxyaw
    yawtw = yaww

    config = Config(minx, miny, minyaw, minyawt, maxx, maxy,
                    maxyaw, maxyawt, xw, yw, yaww, yawtw, xyreso, yawreso)

    return config
