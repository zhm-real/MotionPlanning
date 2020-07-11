import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

import astar
import rs_path as rs
import lib
import trailerlib

PI = np.pi

XY_RESO = 2.0  # [m]
YAW_RESO = np.deg2rad(15.0)  # [rad]
GOAL_TYAW_TH = np.deg2rad(5.0)  # [rad]
Motion_RESO = 0.1  # [m] path interporate resolution
N_STEER = 20.0  # number of steer command
EXTEND_LEN = 5.0  # [m] map extend length
SKIP_COLLISION_CHECK = 4  # skip number for collision check

SWITCH_BACK_COST = 100.0  # switch back penalty cost
BACKWARD_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_ANGLE_COST = 1.0  # steer angle penalty cost
JACKKNIF_COST = 200.0  # Jackknif cost
H_COST = 5.0  # Heuristic cost

WB = lib.WB  # [m] Wheel base
LT = lib.LT  # [m] length of trailer
MAX_STEER = lib.MAX_STEER  # [rad] maximum steering angle


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

    syaw, gyaw = rs.pi_2_pi(syaw), rs.pi_2_pi(gyaw)

    obs = [[x, y] for x, y in zip(ox, oy)]
    kdtree = kd.KDTree(obs)

    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)
    n_start = Node(round(sx / xyreso), round(sy / xyreso), round(syaw / yawreso),
                   1, [sx], [sy], [syaw], [syawt], [1], 0.0, 0.0, -1)
    n_goal = Node(round(gx / xyreso), round(gy / xyreso), round(gyaw / yawreso),
                  1, [gx], [gy], [gyaw], [gyawt], [1], 0.0, 0.0, -1)
    heuristic_map = astar.calc_holonomic_with_obs_heuristic(n_goal, P.ox, P.oy, P.xyreso, 1.0)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_start, P)] = n_start

    fnode = None

    q_priority = []
    heapq.heappush(q_priority,
                   (hybrid_cost(n_start, heuristic_map, P), calc_index(n_start, P)))

    steer, direct = get_motion()

    while True:
        if not open_set:
            return None

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        isupdated, fpath = update_node_with_analystic_expantion(n_curr, n_goal, P, gyawt)

        if isupdated:
            fnode = fpath
            break

        inityawt = n_curr.yawt[0]

        for i in range(len(steer)):
            node = calc_next_node(n_curr, ind, steer[i], direct[i], P)

            if not verify_index(node, P, ox, oy, inityawt, kdtree):
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                heapq.heappush(q_priority, (hybrid_cost(node, heuristic_map, P), node_ind))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node

        print("final expand node: ", len(open_set) + len(closed_set))

    path = get_final_path(closed_set, fnode, n_start, P)

    return path


def verify_index(node, c, ox, oy, inityaw1, kdtree):
    # overflow map
    if (node.xind - c.minx) >= c.xw:
        return False
    elif (node.xind - c.minx) <= 0:
        return False

    if (node.yind - c.miny) >= c.yw:
        return False
    elif (node.yind - c.miny) <= 0:
        return False

    # check collisiton
    steps = [Motion_RESO * x for x in node.directions]
    yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(node.x, node.y, node.yaw, inityaw1, steps)

    ind = range(0, len(node.x), SKIP_COLLISION_CHECK)

    nodex, nodey, nodeyaw, nodeyaw1 = [], [], [], []

    for k in ind:
        nodex.append(node.x[k])
        nodey.append(node.y[k])
        nodeyaw.append(node.yaw[k])
        nodeyaw1.append(yaw1[k])

    if not trailerlib.check_trailer_collision(ox, oy, nodex, nodey, nodeyaw, nodeyaw1, kdtree=kdtree):
        return False

    return True


def calc_next_node(n_curr, ind, u, d, P):
    arc_l = XY_RESO * 1.5

    nlist = int(arc_l / Motion_RESO) + 1
    xlist = [0.0 for _ in range(nlist)]
    ylist = [0.0 for _ in range(nlist)]
    yawlist = [0.0 for _ in range(nlist)]
    yawtlist = [0.0 for _ in range(nlist)]

    xlist[0] = n_curr.x[-1] + d * Motion_RESO * math.cos(n_curr.yaw[-1])
    ylist[0] = n_curr.y[-1] + d * Motion_RESO * math.sin(n_curr.yaw[-1])
    yawlist[0] = rs.pi_2_pi(n_curr.yaw[-1] + d * Motion_RESO / WB * math.tan(u))
    yawtlist[0] = rs.pi_2_pi(
        n_curr.yawt[-1] + d * Motion_RESO / LT * math.sin(n_curr.yaw[-1] - n_curr.yawt[-1]))

    for i in range(nlist - 1):
        xlist[i + 1] = xlist[i] + d * Motion_RESO * math.cos(yawlist[i])
        ylist[i + 1] = ylist[i] + d * Motion_RESO * math.sin(yawlist[i])
        yawlist[i + 1] = rs.pi_2_pi(yawlist[i] + d * Motion_RESO / WB * math.tan(u))
        yawtlist[i + 1] = rs.pi_2_pi(yawtlist[i] + d * Motion_RESO / LT * math.sin(yawlist[i] - yawtlist[i]))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    addedcost = 0.0

    if d > 0:
        direction = True
        addedcost += abs(arc_l)
    else:
        direction = False
        addedcost += abs(arc_l) * BACKWARD_COST

    # swich back penalty
    if direction != n_curr.direction:  # switch back penalty
        addedcost += SWITCH_BACK_COST

    # steer penalyty
    addedcost += STEER_ANGLE_COST * abs(u)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(n_curr.steer - u)

    # jacknif cost
    addedcost += JACKKNIF_COST * sum([abs(rs.pi_2_pi(x - y)) for x, y in zip(yawlist, yawtlist)])

    cost = n_curr.cost + addedcost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, yawtlist, directions, u, cost, ind)

    return node


def is_same_grid(node1, node2):
    if node1.xind != node2.xind:
        return False

    if node1.yind != node2.yind:
        return False

    if node1.yawind != node2.yawind:
        return False

    return True


def update_node_with_analystic_expantion(n_curr, n_goal, P, gyawt):
    apath = analystic_expantion(n_curr, n_goal, P)

    if apath:
        fx = apath.x[1:len(apath.x)]
        fy = apath.y[1:len(apath.y)]
        fyaw = apath.yaw[1:len(apath.yaw)]
        steps = [Motion_RESO * x for x in apath.directions]
        yaw1 = lib.calc_trailer_yaw_from_xyyaw(apath.x, apath.y, apath.yaw, n_curr.yawt[-1], steps)
        if abs(rs.pi_2_pi(yaw1[-1] - gyawt)) >= GOAL_TYAW_TH:
            return False, None  # no update

        fcost = n_curr.cost + calc_rs_path_cost(apath, yaw1)
        fyaw1 = P.yaw1[1:len(yaw1)]
        fpind = calc_index(n_curr, P)

        fd = []
        for d in apath.directions[1:len(apath.directions)]:
            if d >= 0:
                fd.append(True)
            else:
                fd.append(False)

        fsteer = 0.0

        fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                     fx, fy, fyaw, fyaw1, fd, fsteer, fcost, fpind)

        return True, fpath

    return False, None


def analystic_expantion(n_curr, n_goal, P):
    sx, sy, syaw, syawt = n_curr.x[-1], n_curr.y[-1], n_curr.yaw[-1], n_curr.yawt[-1]
    gx, gy, gyaw, gyawt = n_goal.x[-1], n_goal.y[-1], n_goal.yaw[-1], n_goal.yawt[-1]

    maxc = math.tan(MAX_STEER) / lib.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=Motion_RESO)

    if not paths:
        return None

    path_priority = []
    for path in paths:
        steps = [Motion_RESO * x for x in path.directions]
        yawt = lib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, syawt, steps)
        heapq.heappush(path_priority, (calc_rs_path_cost(path, yawt), path))

    _, path = heapq.heappop(path_priority)
    steps = [Motion_RESO * x for x in path.directions]
    yawt = lib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, syawt, steps)
    ind = range(0, len(path.x), SKIP_COLLISION_CHECK)

    pathx, pathy, pathyaw, pathyawt = [], [], [], []

    for k in ind:
        pathx.append(path.x[k])
        pathy.append(path.y[k])
        pathyaw.append(path.yaw[k])
        pathyawt.append(yawt[k])

    if trailerlib.check_trailer_collision(P.ox, P.oy, pathx, pathy, pathyaw, pathyawt, kdtree=P.kdtree):
        return path

    return None


def calc_rs_path_cost(rspath, yawt):
    cost = 0.0
    for length in rspath.lengths:
        if length >= 0:
            cost += 1
        else:
            cost += abs(length) * BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += SWITCH_BACK_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += STEER_ANGLE_COST * abs(MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    cost += JACKKNIF_COST * sum([abs(rs.pi_2_pi(x - y)) for x, y in zip(rspath.yaw, yawt)])

    return cost


def get_motion():
    elem = [i for i in np.arange(MAX_STEER / N_STEER, MAX_STEER, MAX_STEER / N_STEER)]
    steer = [0.0] + [i for i in elem] + [-i for i in elem]
    direct = [1.0 for _ in steer] + [-1.0 for _ in steer]
    steer += steer

    return steer, direct


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    yawt_ind = round(node.yawt[-1] / P.yawreso)
    ind += (yawt_ind - P.minyawt) * P.xw * P.yw * P.yaww

    return ind


def hybrid_cost(node, hmap, P):
    cost = node.cost + H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def get_final_path(closed, ngoal, nstart, P):
    rx = list(reversed(ngoal.x))
    ry = list(reversed(ngoal.y))
    ryaw = list(reversed(ngoal.yaw))
    ryaw1 = list(reversed(ngoal.yaw1))
    direction = list(reversed(ngoal.directions))
    nid = ngoal.pind
    finalcost = ngoal.cost

    while True:
        n = closed[nid]
        rx = rx + list(reversed(n.x))
        ry = ry + list(reversed(n.y))
        ryaw = ryaw + list(reversed(n.yaw))
        ryaw1 = ryaw1 + list(reversed(n.yaw1))
        direction = direction + list(reversed(n.directions))
        nid = n.pind

        if is_same_grid(n, nstart):
            break

    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    ryaw1 = list(reversed(ryaw1))
    direction = list(reversed(direction))

    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, ryaw1, direction, finalcost)

    return path


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minxr = (min(ox) - EXTEND_LEN)
    minyr = (min(oy) - EXTEND_LEN)
    maxxr = (max(ox) - EXTEND_LEN)
    maxyr = (max(oy) - EXTEND_LEN)

    minx = round(minxr / xyreso)
    miny = round(minyr / xyreso)
    maxx = round(maxxr / xyreso)
    maxy = round(maxyr / xyreso)

    ox.append(minxr)
    ox.append(maxxr)
    oy.append(minyr)
    oy.append(maxyr)

    xw, yw = maxx - minx, maxy - miny

    minyaw, maxyaw = round(-PI / yawreso) - 1, round(PI / yawreso)
    yaww = maxyaw - minyaw

    minyawt, maxyawt, yawtw = minyaw, maxyaw, yaww

    P = Para(minx, miny, minyaw, minyawt, maxx, maxy, maxyaw, maxyawt,
             xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree)

    return P


def main():
    print("start!")

    sx = 14.0  # [m]
    sy = 10.0  # [m]
    syaw0 = np.deg2rad(00.0)
    syaw1 = np.deg2rad(00.0)

    gx = 0.0  # [m]
    gy = 0.0  # [m]
    gyaw0 = np.deg2rad(90.0)
    gyaw1 = np.deg2rad(90.0)

    ox, oy = [], []

    for i in range(-25, -23):
        ox.append(float(i))
        oy.append(15.0)

    for i in range(0, 3):
        ox.append(float(i))
        oy.append(15.0)

    for i in range(23, 25):
        ox.append(float(i))
        oy.append(15.0)
    #
    for i in range(-25, -23):
        ox.append(float(i))
        oy.append(4.0)

    for i in range(-6, -4):
        ox.append(float(i))
        oy.append(4.0)
    #
    # for i in range(-15, 5):
    #     ox.append(-4.0)
    #     oy.append(float(i))
    #
    # for i in range(-15, 5):
    #     ox.append(4.0)
    #     oy.append(float(i))

    oox = ox[:]
    ooy = oy[:]

    path = hybrid_astar_planning(sx, sy, syaw0, syaw1, gx, gy, gyaw0,
                                 gyaw1, ox, oy, XY_RESO, YAW_RESO)

    plt.plot(oox, ooy, ".k")
    trailerlib.plot_trailer(sx, sy, syaw0, syaw1, 0.0)
    trailerlib.plot_trailer(gx, gy, gyaw0, gyaw1, 0.0)
    x = path.x
    y = path.y
    yaw = path.yaw
    yaw1 = path.yaw1
    direction = path.direction

    steer = 0.0
    for ii in range(len(x)):
        plt.cla()
        plt.plot(oox, ooy, ".k")
        plt.plot(x, y, "-r", label="Hybrid A* path")

        if ii < len(x) - 2:
            k = (yaw[ii + 1] - yaw[ii]) / Motion_RESO
            if ~direction[ii]:
                k *= -1
            steer = math.atan2(WB * k, 1.0)
        else:
            steer = 0.0
        trailerlib.plot_trailer(gx, gy, gyaw0, gyaw1, 0.0)
        trailerlib.plot_trailer(x[ii], y[ii], yaw[ii], yaw1[ii], steer)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

    print("Done")
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
