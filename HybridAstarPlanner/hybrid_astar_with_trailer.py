"""
Hybrid A* with trailer
@author: Huiming Zhou
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

import HybridAstarPlanner.astar as astar
import HybridAstarPlanner.draw as draw
import CurvesGenerator.reeds_shepp as rs


class C:  # Parameter config
    PI = np.pi

    XY_RESO = 2.0  # [m]
    YAW_RESO = np.deg2rad(15.0)  # [rad]
    GOAL_YAW_ERROR = np.deg2rad(3.0)  # [rad]
    MOVE_STEP = 0.2  # [m] path interporate resolution
    N_STEER = 20.0  # number of steer command
    COLLISION_CHECK_STEP = 10  # skip number for collision check
    EXTEND_AREA = 5.0  # [m] map extend length

    GEAR_COST = 100.0  # switch back penalty cost
    BACKWARD_COST = 5.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost
    SCISSORS_COST = 200.0  # scissors cost
    H_COST = 10.0  # Heuristic cost

    W = 3.0  # [m] width of vehicle
    WB = 3.5  # [m] wheel base: rear to front steer
    WD = 0.7 * W  # [m] distance between left-right wheels
    RF = 4.5  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0  # [m] distance from rear to vehicle back end of vehicle

    RTR = 8.0  # [m] rear to trailer wheel
    RTF = 1.0  # [m] distance from rear to vehicle front end of trailer
    RTB = 9.0  # [m] distance from rear to vehicle back end of trailer
    TR = 0.5  # [m] tyre radius
    TW = 1.0  # [m] tyre width
    MAX_STEER = 0.6  # [rad] maximum steering angle


class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, yawt, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, minyawt, maxx, maxy, maxyaw, maxyawt,
                 xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree):
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
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, yawt, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.direction = direction
        self.cost = cost


class QueuePrior:
    def __init__(self):
        self.queue = []

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, item))  # reorder x using priority

    def get(self):
        return heapq.heappop(self.queue)[1]  # pop out element with smallest priority


def hybrid_astar_planning(sx, sy, syaw, syawt, gx, gy,
                          gyaw, gyawt, ox, oy, xyreso, yawreso):
    """
    planning hybrid A* path.
    :param sx: starting node x position [m]
    :param sy: starting node y position [m]
    :param syaw: starting node yaw angle [rad]
    :param syawt: starting node trailer yaw angle [rad]
    :param gx: goal node x position [m]
    :param gy: goal node y position [m]
    :param gyaw: goal node yaw angle [rad]
    :param gyawt: goal node trailer yaw angle [rad]
    :param ox: obstacle x positions [m]
    :param oy: obstacle y positions [m]
    :param xyreso: grid resolution [m]
    :param yawreso: yaw resolution [m]
    :return: hybrid A* path
    """

    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [syawt], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [gyawt], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    while True:
        if not open_set:
            return None

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P)

        if update:
            fnode = fpath
            break

        yawt0 = n_curr.yawt[0]

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not is_index_ok(node, yawt0, P):
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node

    print("final expand node: ", len(open_set) + len(closed_set))

    return extract_path(closed_set, fnode, nstart)


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, ryawt, direc = [], [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        ryawt += node.yawt[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    ryawt = ryawt[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, ryawt, direc, cost)

    return path


def update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P):
    path = analystic_expantion(n_curr, ngoal, P)  # rs path: n -> ngoal

    if not path:
        return False, None

    steps = [C.MOVE_STEP * d for d in path.directions]
    yawt = calc_trailer_yaw(path.yaw, n_curr.yawt[-1], steps)

    if abs(rs.pi_2_pi(yawt[-1] - gyawt)) >= C.GOAL_YAW_ERROR:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]

    fd = []
    for d in path.directions[1:-1]:
        if d >= 0:
            fd.append(1.0)
        else:
            fd.append(-1.0)
    # fd = path.directions[1:-1]

    fcost = n_curr.cost + calc_rs_path_cost(path, yawt)
    fpind = calc_index(n_curr, P)
    fyawt = yawt[1:-1]
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fyawt, fd, fsteer, fcost, fpind)

    return True, fpath


def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    if not paths:
        return None

    pq = QueuePrior()
    for path in paths:
        steps = [C.MOVE_STEP * d for d in path.directions]
        yawt = calc_trailer_yaw(path.yaw, node.yawt[-1], steps)
        pq.put(path, calc_rs_path_cost(path, yawt))

    # while not pq.empty():
    path = pq.get()
    steps = [C.MOVE_STEP * d for d in path.directions]
    yawt = calc_trailer_yaw(path.yaw, node.yawt[-1], steps)
    ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

    pathx = [path.x[k] for k in ind]
    pathy = [path.y[k] for k in ind]
    pathyaw = [path.yaw[k] for k in ind]
    pathyawt = [yawt[k] for k in ind]

    if not is_collision(pathx, pathy, pathyaw, pathyawt, P):
        return path

    return None


def calc_next_node(n, ind, u, d, P):
    step = C.XY_RESO * 2.0

    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n.x[-1] + d * C.MOVE_STEP * math.cos(n.yaw[-1])]
    ylist = [n.y[-1] + d * C.MOVE_STEP * math.sin(n.yaw[-1])]
    yawlist = [rs.pi_2_pi(n.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]
    yawtlist = [rs.pi_2_pi(n.yawt[-1] +
                           d * C.MOVE_STEP / C.RTR * math.sin(n.yaw[-1] - n.yawt[-1]))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))
        yawtlist.append(rs.pi_2_pi(yawtlist[i] +
                                   d * C.MOVE_STEP / C.RTR * math.sin(yawlist[i] - yawtlist[i])))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    cost = 0.0

    if d > 0:
        direction = 1.0
        cost += abs(step)
    else:
        direction = -1.0
        cost += abs(step) * C.BACKWARD_COST

    if direction != n.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.STEER_ANGLE_COST * abs(u)  # steer penalyty
    cost += C.STEER_CHANGE_COST * abs(n.steer - u)  # steer change penalty
    cost += C.SCISSORS_COST * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(yawlist, yawtlist)])  # jacknif cost
    cost = n.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, yawtlist, directions, u, cost, ind)

    return node


def is_collision(x, y, yaw, yawt, P):
    for ix, iy, iyaw, iyawt in zip(x, y, yaw, yawt):
        d = 0.5
        deltal = (C.RTF - C.RTB) / 2.0
        rt = (C.RTF + C.RTB) / 2.0 + d

        ctx = ix + deltal * math.cos(iyawt)
        cty = iy + deltal * math.sin(iyawt)

        idst = P.kdtree.query_ball_point([ctx, cty], rt)

        if idst:
            for i in idst:
                xot = P.ox[i] - ctx
                yot = P.oy[i] - cty

                dx_trail = xot * math.cos(iyawt) + yot * math.sin(iyawt)
                dy_trail = -xot * math.sin(iyawt) + yot * math.cos(iyawt)

                if abs(dx_trail) <= rt and \
                        abs(dy_trail) <= C.W / 2.0 + d:
                    return True

        deltal = (C.RF - C.RB) / 2.0
        rc = (C.RF + C.RB) / 2.0 + d

        cx = ix + deltal * math.cos(iyaw)
        cy = iy + deltal * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], rc)

        if ids:
            for i in ids:
                xo = P.ox[i] - cx
                yo = P.oy[i] - cy

                dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                if abs(dx_car) <= rc and \
                        abs(dy_car) <= C.W / 2.0 + d:
                    return True
        #
        # theta = np.linspace(0, 2 * np.pi, 200)
        # x1 = ctx + np.cos(theta) * rt
        # y1 = cty + np.sin(theta) * rt
        # x2 = cx + np.cos(theta) * rc
        # y2 = cy + np.sin(theta) * rc
        #
        # plt.plot(x1, y1, 'b')
        # plt.plot(x2, y2, 'g')

    return False


def calc_trailer_yaw(yaw, yawt0, steps):
    yawt = [0.0 for _ in range(len(yaw))]
    yawt[0] = yawt0

    for i in range(1, len(yaw)):
        yawt[i] += yawt[i - 1] + steps[i - 1] / C.RTR * math.sin(yaw[i - 1] - yawt[i - 1])

    return yawt


def trailer_motion_model(x, y, yaw, yawt, D, d, L, delta):
    x += D * math.cos(yaw)
    y += D * math.sin(yaw)
    yaw += D / L * math.tan(delta)
    yawt += D / d * math.sin(yaw - yawt)

    return x, y, yaw, yawt


def calc_rs_path_cost(rspath, yawt):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    cost += C.SCISSORS_COST * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(rspath.yaw, yawt)])

    return cost


def calc_motion_set():
    s = [i for i in np.arange(C.MAX_STEER / C.N_STEER,
                              C.MAX_STEER, C.MAX_STEER / C.N_STEER)]

    steer = [0.0] + s + [-i for i in s]
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    yawt_ind = round(node.yawt[-1] / P.yawreso)
    ind += (yawt_ind - P.minyawt) * P.xw * P.yw * P.yaww

    return ind


def is_index_ok(node, yawt0, P):
    if node.xind <= P.minx or \
            node.xind >= P.maxx or \
            node.yind <= P.miny or \
            node.yind >= P.maxy:
        return False

    steps = [C.MOVE_STEP * d for d in node.directions]
    yawt = calc_trailer_yaw(node.yaw, yawt0, steps)

    ind = range(0, len(node.x), C.COLLISION_CHECK_STEP)

    x = [node.x[k] for k in ind]
    y = [node.y[k] for k in ind]
    yaw = [node.yaw[k] for k in ind]
    yawt = [yawt[k] for k in ind]

    if is_collision(x, y, yaw, yawt, P):
        return False

    return True


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minxm = min(ox) - C.EXTEND_AREA
    minym = min(oy) - C.EXTEND_AREA
    maxxm = max(ox) + C.EXTEND_AREA
    maxym = max(oy) + C.EXTEND_AREA

    ox.append(minxm)
    oy.append(minym)
    ox.append(maxxm)
    oy.append(maxym)

    minx = round(minxm / xyreso)
    miny = round(minym / xyreso)
    maxx = round(maxxm / xyreso)
    maxy = round(maxym / xyreso)

    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    minyawt, maxyawt, yawtw = minyaw, maxyaw, yaww

    P = Para(minx, miny, minyaw, minyawt, maxx, maxy, maxyaw,
             maxyawt, xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree)

    return P


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def draw_model(x, y, yaw, yawt, steer, color='black'):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    trail = np.array([[-C.RTB, -C.RTB, C.RTF, C.RTF, -C.RTB],
                      [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()
    rltWheel = wheel.copy()
    rrtWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), -math.sin(steer)],
                     [math.sin(steer), math.cos(steer)]])

    Rot3 = np.array([[math.cos(yawt), -math.sin(yawt)],
                     [math.sin(yawt), math.cos(yawt)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    rltWheel += np.array([[-C.RTR], [C.WD / 2]])
    rrtWheel += np.array([[-C.RTR], [-C.WD / 2]])

    rltWheel = np.dot(Rot3, rltWheel)
    rrtWheel = np.dot(Rot3, rrtWheel)
    trail = np.dot(Rot3, trail)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    rrtWheel += np.array([[x], [y]])
    rltWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])
    trail += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(trail[0, :], trail[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    plt.plot(rrtWheel[0, :], rrtWheel[1, :], color)
    plt.plot(rltWheel[0, :], rltWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)


def design_obstacles():
    ox, oy = [], []

    for i in range(-30, 31):
        ox.append(i)
        oy.append(38)

    for i in range(-30, -6):
        ox.append(i)
        oy.append(23)

    for i in range(7, 31):
        ox.append(i)
        oy.append(23)

    for j in range(0, 24):
        ox.append(-6)
        oy.append(j)

    for j in range(0, 24):
        ox.append(6)
        oy.append(j)

    for i in range(-6, 7):
        ox.append(i)
        oy.append(0)

    return ox, oy


def test(x, y, yaw, yawt, ox, oy):
    d = 0.5
    deltal = (C.RTF - C.RTB) / 2.0
    rt = (C.RTF + C.RTB) / 2.0 + d

    ctx = x + deltal * math.cos(yawt)
    cty = y + deltal * math.sin(yawt)

    deltal = (C.RF - C.RB) / 2.0
    rc = (C.RF + C.RB) / 2.0 + d

    xot = ox - ctx
    yot = oy - cty

    dx_trail = xot * math.cos(yawt) + yot * math.sin(yawt)
    dy_trail = -xot * math.sin(yawt) + yot * math.cos(yawt)

    if abs(dx_trail) <= rt - d and \
            abs(dy_trail) <= C.W / 2.0:
        print("test1: Collision")
    else:
        print("test1: No collision")

    # test 2

    cx = x + deltal * math.cos(yaw)
    cy = y + deltal * math.sin(yaw)

    xo = ox - cx
    yo = oy - cy

    dx_car = xo * math.cos(yaw) + yo * math.sin(yaw)
    dy_car = -xo * math.sin(yaw) + yo * math.cos(yaw)

    if abs(dx_car) <= rc - d and \
            abs(dy_car) <= C.W / 2.0:
        print("test2: Collision")
    else:
        print("test2: No collision")

    theta = np.linspace(0, 2 * np.pi, 200)
    x1 = ctx + np.cos(theta) * rt
    y1 = cty + np.sin(theta) * rt
    x2 = cx + np.cos(theta) * rc
    y2 = cy + np.sin(theta) * rc

    plt.plot(x1, y1, 'b')
    plt.plot(x2, y2, 'g')
    plt.plot(ox, oy, 'sr')

    plt.plot([-rc, -rc, rc, rc, -rc],
             [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2])
    plt.plot([-rt, -rt, rt, rt, -rt],
             [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2])
    plt.plot(dx_car, dy_car, 'sr')
    plt.plot(dx_trail, dy_trail, 'sg')


def main():
    print("start!")

    sx, sy = 18.0, 34.0  # [m]
    syaw0 = np.deg2rad(180.0)
    syawt = np.deg2rad(180.0)

    gx, gy = 0.0, 12.0  # [m]
    gyaw0 = np.deg2rad(90.0)
    gyawt = np.deg2rad(90.0)

    ox, oy = design_obstacles()
    plt.plot(ox, oy, 'sk')
    draw_model(sx, sy, syaw0, syawt, 0.0)
    draw_model(gx, gy, gyaw0, gyawt, 0.0)
    # test(sx, sy, syaw0, syawt, 3.5, 32)
    # plt.axis("equal")
    # plt.show()

    oox, ooy = ox[:], oy[:]

    t0 = time.time()
    path = hybrid_astar_planning(sx, sy, syaw0, syawt, gx, gy, gyaw0, gyawt,
                                 oox, ooy, C.XY_RESO, C.YAW_RESO)
    t1 = time.time()
    print("running T: ", t1 - t0)

    x = path.x
    y = path.y
    yaw = path.yaw
    yawt = path.yawt
    direction = path.direction

    plt.pause(10)

    for k in range(len(x)):
        plt.cla()
        plt.plot(ox, oy, "sk")
        plt.plot(x, y, linewidth=1.5, color='r')

        if k < len(x) - 2:
            dy = (yaw[k + 1] - yaw[k]) / C.MOVE_STEP
            steer = rs.pi_2_pi(math.atan(C.WB * dy / direction[k]))
        else:
            steer = 0.0

        draw_model(gx, gy, gyaw0, gyawt, 0.0, 'gray')
        draw_model(x[k], y[k], yaw[k], yawt[k], steer)
        plt.axis("equal")
        plt.pause(0.0001)

    plt.show()
    print("Done")


if __name__ == '__main__':
    main()
