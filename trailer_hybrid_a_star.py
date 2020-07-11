import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
import scipy.spatial.kdtree as KD

import astar
import rs_path
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

WB = trailerlib.WB  # [m] Wheel base
LT = trailerlib.LT  # [m] length of trailer
MAX_STEER = trailerlib.MAX_STEER  # [rad] maximum steering angle


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
    def __init__(self, x, y, yaw, yawt, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.direction = direction
        self.cost = cost


def calc_hybrid_astar_path(sx, sy, syaw, syawt, gx, gy, gyaw, gyawt, ox, oy, xyreso, yawreso):
    """
    sx: start x position [m]
    sy: start y position [m]
    syaw: start yaw angle [rad]
    syawt: start trailer yaw angle [m]
    gx: goal x position [m]
    gx: goal x position [m]
    gyaw: goal yaw angle [m]
    gyawt: goal trailer yaw angle [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xyreso: grid resolution [m]
    yawreso: yaw angle resolution [rad]
    """

    syaw = rs_path.pi_2_pi(syaw)
    gyaw = rs_path.pi_2_pi(gyaw)

    obs = np.array([[x, y] for x, y in zip(ox, oy)])
    kdtree = KD.KDTree(obs)

    c = get_config(ox, oy, xyreso, yawreso)

    nstart = Node(round(sx/xyreso), round(sy/xyreso), round(syaw/yawreso),
                  True, [sx], [sy], [syaw], [syawt], [True], 0.0, 0.0, -1)
    ngoal = Node(round(gx/xyreso), round(gy/xyreso), round(gyaw/yawreso),
                 True, [gx], [gy], [gyaw], [gyawt], [True], 0.0, 0.0, -1)

    h_dp = astar.calc_holonomic_with_obs_heuristic(ngoal, ox, oy, xyreso, 1.0)  # cost of each node

    openset, closed_set = dict(), dict()
    openset[calc_index(nstart, c)] = nstart

    fnode = None

    pq = []
    heapq.heappush(pq, (get_cost(nstart, h_dp, ngoal, c), calc_index(nstart, c)))

    u, d = get_motion_inputs()
    nmotion = len(u)

    while True:
        if len(openset) == 0:
            print("Error: Cannot find path, No open set")
            return []

        _, c_id = heapq.heappop(pq)
        current = openset[c_id]

        openset.pop(c_id)
        closed_set[c_id] = current

        isupdated, fpath = update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree, gyawt)

        if isupdated:
            fnode = fpath
            break

        inityawt = current.yawt[0]

        for i in range(nmotion):
            node = calc_next_node(current, c_id, u[i], d[i], c)

            if not verify_index(node, c, ox, oy, inityawt, kdtree):
                continue

            node_ind = calc_index(node, c)

            if node_ind in closed_set:
                continue

            if node_ind not in openset:
                openset[node_ind] = node
                heapq.heappush(pq, (get_cost(node, h_dp, ngoal, c), node_ind))
            else:
                if openset[node_ind].cost > node.cost:
                    openset[node_ind] = node

    print("final expand node: ", len(openset) + len(closed_set))

    path = get_final_path(closed_set, fnode, nstart, c)

    return path


def update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree, gyawt):
    apath = analystic_expantion(current, ngoal, c, ox, oy, kdtree)

    if apath:
        fx = apath.x[1:len(apath.x)]
        fy = apath.y[1:len(apath.y)]
        fyaw = apath.yaw[1:len(apath.yaw)]
        steps = [Motion_RESO * x for x in apath.directions]
        yawt = trailerlib.calc_trailer_yaw_from_xyyaw(apath.x, apath.y, apath.yaw, current.yawt[-1], steps)
        if abs(rs_path.pi_2_pi(yawt[-1] - gyawt)) >= GOAL_TYAW_TH:
            return False, None  # no update

        fcost = current.cost + calc_rs_path_cost(apath, yawt)
        fyawt = yawt[1:len(yawt)]
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:len(apath.directions)]:
            if d >= 0:
                fd.append(True)
            else:
                fd.append(False)

        fsteer = 0.0

        fpath = Node(current.xind, current.yind, current.yawind,
                     current.direction, fx, fy, fyaw, fyawt, fd, fsteer, fcost, fpind)

        return True, fpath

    return False, None


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

    cost += JACKKNIF_COST * sum([abs(rs_path.pi_2_pi(x - y)) for x, y in zip(rspath.yaw, yawt)])

    return cost


def analystic_expantion(n, ngoal, c, ox, oy, kdtree):
    sx = n.x[-1]
    sy = n.y[-1]
    syaw = n.yaw[-1]

    max_curvature = math.tan(MAX_STEER)/WB
    paths = rs_path.calc_all_paths(sx, sy, syaw, ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1],
                                   max_curvature, step_size=Motion_RESO)
    if len(paths) == 0:
        return None

    pathqueue = []
    for path in paths:
        steps = [Motion_RESO * x for x in path.directions]
        yawt = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yawt[-1], steps)
        heapq.heappush(pathqueue, (calc_rs_path_cost(path, yawt), path))

    # for i in range(len(pathqueue)):
    _, path = heapq.heappop(pathqueue)

    steps = [Motion_RESO * x for x in path.directions]
    yawt = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yawt[-1], steps)
    ind = range(0, len(path.x), SKIP_COLLISION_CHECK)

    pathx, pathy, pathyaw, pathyawt = [], [], [], []

    for k in ind:
        pathx.append(path.x[k])
        pathy.append(path.y[k])
        pathyaw.append(path.yaw[k])
        pathyawt.append(yawt[k])

    if trailerlib.check_trailer_collision(ox, oy, pathx, pathy, pathyaw, pathyawt, kdtree=kdtree):
        return path

    return None


def get_motion_inputs():
    up = [i for i in np.arange(MAX_STEER / N_STEER, MAX_STEER, MAX_STEER / N_STEER)]
    u = [0.0] + [i for i in up] + [-i for i in up]
    d = [1.0 for _ in range(len(u))] + [-1.0 for _ in range(len(u))]
    u = u + u

    return u, d


def verify_index(node, c, ox, oy, inityawt, kdtree):
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
    yawt = trailerlib.calc_trailer_yaw_from_xyyaw(node.x, node.y, node.yaw, inityawt, steps)

    ind = range(0, len(node.x), SKIP_COLLISION_CHECK)

    nodex, nodey, nodeyaw, nodeyawt = [], [], [], []

    for k in ind:
        nodex.append(node.x[k])
        nodey.append(node.y[k])
        nodeyaw.append(node.yaw[k])
        nodeyawt.append(yawt[k])

    if not trailerlib.check_trailer_collision(ox, oy, nodex, nodey, nodeyaw, nodeyawt, kdtree=kdtree):
        return False

    return True


def calc_next_node(current, c_id, u, d, c):
    arc_l = XY_RESO * 1.5

    nlist = int(arc_l / Motion_RESO) + 1
    xlist = [0.0 for _ in range(nlist)]
    ylist = [0.0 for _ in range(nlist)]
    yawlist = [0.0 for _ in range(nlist)]
    yawtlist = [0.0 for _ in range(nlist)]

    xlist[0] = current.x[-1] + d * Motion_RESO * math.cos(current.yaw[-1])
    ylist[0] = current.y[-1] + d * Motion_RESO * math.sin(current.yaw[-1])
    yawlist[0] = rs_path.pi_2_pi(current.yaw[-1] + d * Motion_RESO / WB * math.tan(u))
    yawtlist[0] = rs_path.pi_2_pi(
        current.yawt[-1] + d * Motion_RESO / LT * math.sin(current.yaw[-1] - current.yawt[-1]))

    for i in range(nlist - 1):
        xlist[i + 1] = xlist[i] + d * Motion_RESO * math.cos(yawlist[i])
        ylist[i + 1] = ylist[i] + d * Motion_RESO * math.sin(yawlist[i])
        yawlist[i + 1] = rs_path.pi_2_pi(yawlist[i] + d * Motion_RESO / WB * math.tan(u))
        yawtlist[i + 1] = rs_path.pi_2_pi(yawtlist[i] + d * Motion_RESO / LT * math.sin(yawlist[i] - yawtlist[i]))

    xind = round(xlist[-1] / c.xyreso)
    yind = round(ylist[-1] / c.xyreso)
    yawind = round(yawlist[-1] / c.yawreso)

    addedcost = 0.0

    if d > 0:
        direction = True
        addedcost += abs(arc_l)
    else:
        direction = False
        addedcost += abs(arc_l) * BACKWARD_COST

    # swich back penalty
    if direction != current.direction:  # switch back penalty
        addedcost += SWITCH_BACK_COST

    # steer penalyty
    addedcost += STEER_ANGLE_COST * abs(u)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(current.steer - u)

    # jacknif cost
    addedcost += JACKKNIF_COST * sum([abs(rs_path.pi_2_pi(x - y)) for x, y in zip(yawlist, yawtlist)])

    cost = current.cost + addedcost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, yawtlist, directions, u, cost, c_id)

    return node


def is_same_grid(node1, node2):
    if node1.xind != node2.xind:
        return False

    if node1.yind != node2.yind:
        return False

    if node1.yawind != node2.yawind:
        return False

    return True


def calc_index(node, c):
    ind = (node.yawind - c.minyaw) * c.xw * c.yw + (node.yind - c.miny) * c.xw + (node.xind - c.minx)

    # 4D grid
    yawtind = round(node.yawt[-1] / c.yawreso)
    ind += (yawtind - c.minyawt) * c.xw * c.yw * c.yaww

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


def get_config(ox, oy, xyreso, yawreso):
    min_x_m = min(ox) - EXTEND_LEN
    min_y_m = min(oy) - EXTEND_LEN
    max_x_m = max(ox) + EXTEND_LEN
    max_y_m = max(oy) + EXTEND_LEN

    ox.append(min_x_m)
    oy.append(min_y_m)
    ox.append(max_x_m)
    oy.append(max_y_m)

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


def get_final_path(closed, ngoal, nstart, c):
    rx = list(reversed(ngoal.x))
    ry = list(reversed(ngoal.y))
    ryaw = list(reversed(ngoal.yaw))
    ryawt = list(reversed(ngoal.yawt))
    direction = list(reversed(ngoal.directions))
    nid = ngoal.pind
    finalcost = ngoal.cost

    while True:
        n = closed[nid]
        rx = rx + list(reversed(n.x))
        ry = ry + list(reversed(n.y))
        ryaw = ryaw + list(reversed(n.yaw))
        ryawt = ryawt + list(reversed(n.yawt))
        direction = direction + list(reversed(n.directions))
        nid = n.pind

        if is_same_grid(n, nstart):
            break

    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    ryawt = list(reversed(ryawt))
    direction = list(reversed(direction))

    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, ryawt, direction, finalcost)

    return path


def get_cost(n, h_dp, ngoal, c):
    return n.cost + H_COST * h_dp[n.xind - c.minx][n.yind - c.miny]


def main():
    print("start!")

    sx = 14.0  # [m]
    sy = 10.0  # [m]
    syaw0 = np.deg2rad(00.0)
    syawt = np.deg2rad(00.0)

    gx = 0.0  # [m]
    gy = 0.0  # [m]
    gyaw0 = np.deg2rad(90.0)
    gyawt = np.deg2rad(90.0)

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

    path = calc_hybrid_astar_path(sx, sy, syaw0, syawt, gx, gy, gyaw0,
                                  gyawt, ox, oy, XY_RESO, YAW_RESO)

    plt.plot(oox, ooy, ".k")
    trailerlib.plot_trailer(sx, sy, syaw0, syawt, 0.0)
    trailerlib.plot_trailer(gx, gy, gyaw0, gyawt, 0.0)
    x = path.x
    y = path.y
    yaw = path.yaw
    yawt = path.yawt
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
        trailerlib.plot_trailer(gx, gy, gyaw0, gyawt, 0.0)
        trailerlib.plot_trailer(x[ii], y[ii], yaw[ii], yawt[ii], steer)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

    print("Done")
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
