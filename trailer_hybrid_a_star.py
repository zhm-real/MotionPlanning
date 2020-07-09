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
    kdtree = []
    ox, oy = ox[:], oy[:]

    for x, y in zip(ox, oy):
        kdtree.append((x, y))

    c = calc_config(ox, oy, xyreso, yawreso)
    nstart = Node(round(sx/xyreso), round(sy/xyreso), round(syaw/yawreso),
                  True, [sx], [sy], [syaw], [syaw1], [True], 0.0, 0.0, -1)
    ngoal = Node(round(gx/xyreso), round(gy/xyreso), round(gyaw/yawreso),
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
        if len(openset) == 0:
            print("Error: Cannot find path, No open set")
            return []

        _, c_id = heapq.heappop(pq)
        current = openset[c_id]

        openset.pop(c_id)
        closed_set[c_id] = current

        isupdated, fpath = update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree, gyaw1)
        if isupdated:
            fnode = fpath
            break

        inityaw1 = current.yaw1[0]

        for i in range(nmotion):
            node = calc_next_node(current, c_id, u[i], d[i], c)

            if not verify_index(node, c, ox, oy, inityaw1, kdtree):
                continue

            node_ind = calc_index(node, c)

            if node_ind in closed_set:
                continue

            if node_ind not in openset:
                openset[node_ind] = node
                heapq.heappush(pq, (calc_cost(node, h_dp, ngoal, c), node_ind))
            else:
                if openset[node_ind].cost > node.cost:
                    openset[node_ind] = node

    print("final expand node: ", len(openset) + len(closed_set))

    path = get_final_path(closed_set, fnode, nstart, c)

    return path


def update_node_with_analystic_expantion(current, ngoal, c, ox, oy, kdtree, gyaw1):
    apath = analystic_expantion(current, ngoal, c, ox, oy, kdtree)

    if apath:
        fx = apath.x[1:len(apath.x)]
        fy = apath.y[1:len(apath.y)]
        fyaw = apath.yaw[1:len(apath.yaw)]
        steps = [MOTION_RESOLUTION*x for x in apath.directions]
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(apath.x, apath.y, apath.yaw, current.yaw1[-1], steps)
        if abs(rs_path.pi_2_pi(yaw1[-1] - gyaw1)) >= GOAL_TYAW_TH:
            return False, None  # no update

        fcost = current.cost + calc_rs_path_cost(apath, yaw1)
        fyaw1 = yaw1[1:len(yaw1)]
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:len(apath.directions)]:
            if d >= 0:
                fd.append(True)
            else:
                fd.append(False)

        fsteer = 0.0

        fpath = Node(current.xind, current.yind, current.yawind,
                     current.direction, fx, fy, fyaw, fyaw1, fd, fsteer, fcost, fpind)

        return True, fpath

    return False, None


def calc_rs_path_cost(rspath, yaw1):
    cost = 0.0
    for length in rspath.lengths:
        if length >= 0:
            cost += 1
        else:
            cost += abs(length) * BACK_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += SB_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += STEER_COST * abs(MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    cost += JACKKNIF_COST * sum([abs(rs_path.pi_2_pi(x - y)) for x, y in zip(rspath.yaw, yaw1)])

    return cost


def analystic_expantion(n, ngoal, c, ox, oy, kdtree):
    sx = n.x[-1]
    sy = n.y[-1]
    syaw = n.yaw[-1]

    max_curvature = math.tan(MAX_STEER)/WB
    paths = rs_path.calc_all_paths(sx, sy, syaw, ngoal.x[-1], ngoal.y[-1],
                                   ngoal.yaw[-1], max_curvature, step_size=MOTION_RESOLUTION)
    if len(paths) == 0:
        return None

    pathqueue = []
    for path in paths:
        steps = [MOTION_RESOLUTION*x for x in path.directions]
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yaw1[-1], steps)
        heapq.heappush(pathqueue, (calc_rs_path_cost(path, yaw1), path))

    for i in range(len(pathqueue)):
        _, path = heapq.heappop(pathqueue)

        steps = [MOTION_RESOLUTION*x for x in path.directions]
        yaw1 = trailerlib.calc_trailer_yaw_from_xyyaw(path.x, path.y, path.yaw, n.yaw1[-1], steps)
        ind = range(0, len(path.x), SKIP_COLLISION_CHECK)

        pathx, pathy, pathyaw, pathyaw1 = [], [], [], []

        for k in ind:
            pathx.append(path.x[k])
            pathy.append(path.y[k])
            pathyaw.append(path.yaw[k])
            pathyaw1.append(yaw1[k])

        if trailerlib.check_trailer_collision(ox, oy, pathx, pathy, pathyaw, pathyaw1, kdtree=kdtree):
            return path

    return None


def calc_motion_inputs():
    up = [i for i in np.arange(MAX_STEER / N_STEER, MAX_STEER, MAX_STEER / N_STEER)]
    u = [0.0] + [i for i in up] + [-i for i in up]
    d = [1.0 for _ in range(len(u))] + [-1.0 for _ in range(len(u))]
    u = u + u

    return u, d


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
    steps = [MOTION_RESOLUTION*x for x in node.directions]
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


def calc_next_node(current, c_id, u, d, c):
    arc_l = XY_GRID_RESOLUTION * 1.5

    nlist = int(arc_l / MOTION_RESOLUTION) + 1
    xlist = [0.0 for _ in range(nlist)]
    ylist = [0.0 for _ in range(nlist)]
    yawlist = [0.0 for _ in range(nlist)]
    yaw1list = [0.0 for _ in range(nlist)]

    xlist[0] = current.x[-1] + d * MOTION_RESOLUTION * math.cos(current.yaw[-1])
    ylist[0] = current.y[-1] + d * MOTION_RESOLUTION * math.sin(current.yaw[-1])
    yawlist[0] = rs_path.pi_2_pi(current.yaw[-1] + d * MOTION_RESOLUTION / WB * math.tan(u))
    yaw1list[0] = rs_path.pi_2_pi(
        current.yaw1[-1] + d * MOTION_RESOLUTION / LT * math.sin(current.yaw[-1] - current.yaw1[-1]))

    for i in range(nlist - 1):
        xlist[i + 1] = xlist[i] + d * MOTION_RESOLUTION * math.cos(yawlist[i])
        ylist[i + 1] = ylist[i] + d * MOTION_RESOLUTION * math.sin(yawlist[i])
        yawlist[i + 1] = rs_path.pi_2_pi(yawlist[i] + d * MOTION_RESOLUTION / WB * math.tan(u))
        yaw1list[i + 1] = rs_path.pi_2_pi(yaw1list[i] + d * MOTION_RESOLUTION / LT * math.sin(yawlist[i] - yaw1list[i]))

    xind = round(xlist[-1] / c.xyreso)
    yind = round(ylist[-1] / c.xyreso)
    yawind = round(yawlist[-1] / c.yawreso)

    addedcost = 0.0

    if d > 0:
        direction = True
        addedcost += abs(arc_l)
    else:
        direction = False
        addedcost += abs(arc_l) * BACK_COST

    # swich back penalty
    if direction != current.direction:  # switch back penalty
        addedcost += SB_COST

    # steer penalyty
    addedcost += STEER_COST * abs(u)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(current.steer - u)

    # jacknif cost
    addedcost += JACKKNIF_COST * sum([abs(rs_path.pi_2_pi(x - y)) for x, y in zip(yawlist, yaw1list)])

    cost = current.cost + addedcost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist, yawlist, yaw1list, directions, u, cost, c_id)

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


def get_final_path(closed, ngoal, nstart, c):
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


def calc_cost(n, h_dp, ngoal, c):
    return n.cost + H_COST * h_dp[n.xind - c.minx][n.yind - c.miny]


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

    for i in range(-25, 25):
        ox.append(float(i))
        oy.append(15.0)
    #
    for i in range(-25, -4):
        ox.append(float(i))
        oy.append(4.0)

    for i in range(-15, 4):
        ox.append(-4.0)
        oy.append(float(i))

    for i in range(-15, 4):
        ox.append(4.0)
        oy.append(float(i))
    #
    # for i in range(4, 25):
    #     ox.append(float(i))
    #     oy.append(4.0)
    #
    # for i in range(-4, 4):
    #     ox.append(float(i))
    #     oy.append(-15.0)

    oox = ox[:]
    ooy = oy[:]

    path = calc_hybrid_astar_path(sx, sy, syaw0, syaw1, gx, gy, gyaw0,
                                  gyaw1, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

    plt.plot(oox, ooy, ".k")
    trailerlib.plot_trailer(sx, sy, syaw0, syaw1, 0.0)
    trailerlib.plot_trailer(gx, gy, gyaw0, gyaw1, 0.0)
    x = path.x
    y = path.y
    yaw = path.yaw
    yaw1 = path.yaw1
    direction = path.direction

    # fig = plt.plot()
    # steer = 0.0
    # for ii in range(len(x)):
    #     plt.cla()
    plt.plot(oox, ooy, ".k")
    plt.plot(x, y, "-r", label="Hybrid A* path")
    plt.show()


if __name__ == '__main__':
    main()
