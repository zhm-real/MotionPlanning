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

    nstart = Node(round(sx / xyreso), round(sy / xyreso), round(syaw / yawreso),
                  1, [sx], [sy], [syaw], [syawt], [1], 0.0, 0.0, -1)
    ngoal = Node(round(gx / xyreso), round(gy / xyreso), round(gyaw / yawreso),
                 1, [gx], [gy], [gyaw], [gyawt], [1], 0.0, 0.0, -1)

    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, C.VR)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(nstart, P)] = nstart

    fnode = None

    q_priority = []
    heapq.heappush(q_priority,
                   (calc_hybrid_cost(nstart, hmap, P), calc_index(nstart, P)))

    steer, direc = get_motion()

    while True:
        if not open_set:
            return []

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        isupdated, fpath = update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P)

        if isupdated:
            fnode = fpath
            break

        inityawt = n_curr.yawt[0]

        for i in range(len(steer)):
            node = calc_next_node(n_curr, ind, steer[i], direc[i], P)

            if not verify_index(node, inityawt, P):
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                heapq.heappush(q_priority, (calc_hybrid_cost(node, hmap, P), node_ind))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node

    print("final expand node: ", len(open_set) + len(closed_set))

    path = extract_path(closed_set, fnode, nstart)

    return path


def update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P):
    apath = analystic_expantion(n_curr, ngoal, P)   # rs path: n_curr -> ngoal

    if apath:
        fx = apath.x[1:-1]
        fy = apath.y[1:-1]
        fyaw = apath.yaw[1:-1]

        steps = [C.Motion_RESO * d for d in apath.directions]
        yawt = lib.calc_trailer_yaw(apath.yaw, n_curr.yawt[-1], steps)

        if abs(rs.pi_2_pi(yawt[-1] - gyawt)) >= C.GOAL_TYAW_TH:
            return False, None

        fcost = n_curr.cost + calc_rs_path_cost(apath, yawt)
        fyawt = yawt[1:-1]
        fpind = calc_index(n_curr, P)

        fsteer = 0.0
        fd = apath.directions[1:-1]
        fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                     fx, fy, fyaw, fyawt, fd, fsteer, fcost, fpind)

        return True, fpath

    return False, None


def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.Motion_RESO)

    if not paths:
        return None

    pathqueue = []
    for path in paths:
        steps = [C.Motion_RESO * d for d in path.directions]
        yawt = lib.calc_trailer_yaw(path.yaw, node.yawt[-1], steps)
        heapq.heappush(pathqueue, (calc_rs_path_cost(path, yawt), path))

    _, path = heapq.heappop(pathqueue)
    steps = [C.Motion_RESO * d for d in path.directions]
    yawt = lib.calc_trailer_yaw(path.yaw, node.yawt[-1], steps)
    ind = range(0, len(path.x), C.SKIP_COLLISION_CHECK)

    pathx = [path.x[k] for k in ind]
    pathy = [path.y[k] for k in ind]
    pathyaw = [path.yaw[k] for k in ind]
    pathyawt = [yawt[k] for k in ind]

    if lib.check_trailer_collision(pathx, pathy, pathyaw, pathyawt, P):
        return path

    return None


def calc_rs_path_cost(rspath, yawt):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.SWITCH_BACK_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    cost += C.JACKKNIF_COST * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(rspath.yaw, yawt)])

    return cost


def get_motion():
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


def verify_index(node, inityawt, P):
    if node.xind <= P.minx or \
            node.xind >= P.maxx or \
            node.yind <= P.miny or \
            node.yind >= P.maxy:
        return False

    # check collisiton
    # steps = [C.Motion_RESO * x for x in node.directions]
    # yawt = lib.calc_trailer_yaw(node.yaw, inityawt, steps)
    #
    # ind = range(0, len(node.x), C.SKIP_COLLISION_CHECK)
    #
    # nodex = [node.x[k] for k in ind]
    # nodey = [node.y[k] for k in ind]
    # nodeyaw = [node.yaw[k] for k in ind]
    # nodeyawt = [yawt[k] for k in ind]
    #
    # if not lib.check_trailer_collision(nodex, nodey, nodeyaw, nodeyawt, P):
    #     return False

    return True


def calc_next_node(n_curr, c_id, u, d, c):
    arc_l = C.XY_RESO * 1.5

    nlist = int(arc_l / C.Motion_RESO) + 1
    xlist = [0.0 for _ in range(nlist)]
    ylist = [0.0 for _ in range(nlist)]
    yawlist = [0.0 for _ in range(nlist)]
    yawtlist = [0.0 for _ in range(nlist)]

    xlist[0] = n_curr.x[-1] + d * C.Motion_RESO * math.cos(n_curr.yaw[-1])
    ylist[0] = n_curr.y[-1] + d * C.Motion_RESO * math.sin(n_curr.yaw[-1])
    yawlist[0] = rs.pi_2_pi(n_curr.yaw[-1] + d * C.Motion_RESO / C.WB * math.tan(u))
    yawtlist[0] = rs.pi_2_pi(
        n_curr.yawt[-1] + d * C.Motion_RESO / C.LT * math.sin(n_curr.yaw[-1] - n_curr.yawt[-1]))

    for i in range(nlist - 1):
        xlist[i + 1] = xlist[i] + d * C.Motion_RESO * math.cos(yawlist[i])
        ylist[i + 1] = ylist[i] + d * C.Motion_RESO * math.sin(yawlist[i])
        yawlist[i + 1] = rs.pi_2_pi(yawlist[i] + d * C.Motion_RESO / C.WB * math.tan(u))
        yawtlist[i + 1] = rs.pi_2_pi(yawtlist[i] + d * C.Motion_RESO / C.LT * math.sin(yawlist[i] - yawtlist[i]))

    xind = round(xlist[-1] / c.xyreso)
    yind = round(ylist[-1] / c.xyreso)
    yawind = round(yawlist[-1] / c.yawreso)

    addedcost = 0.0

    if d > 0:
        direction = True
        addedcost += abs(arc_l)
    else:
        direction = False
        addedcost += abs(arc_l) * C.BACKWARD_COST

    # swich back penalty
    if direction != n_curr.direction:  # switch back penalty
        addedcost += C.SWITCH_BACK_COST

    # steer penalyty
    addedcost += C.STEER_ANGLE_COST * abs(u)

    # steer change penalty
    addedcost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)

    # jacknif cost
    addedcost += C.JACKKNIF_COST * sum([abs(rs.pi_2_pi(x - y)) for x, y in zip(yawlist, yawtlist)])

    cost = n_curr.cost + addedcost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, yawtlist, directions, u, cost, c_id)

    return node


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minox = min(ox) - C.EXTEND_LEN
    minoy = min(oy) - C.EXTEND_LEN
    maxox = max(ox) + C.EXTEND_LEN
    maxoy = max(oy) + C.EXTEND_LEN

    ox.append(minox)
    oy.append(minoy)
    ox.append(maxox)
    oy.append(maxoy)

    minx = round(minox / xyreso)
    miny = round(minoy / xyreso)
    maxx = round(maxox / xyreso)
    maxy = round(maxoy / xyreso)

    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    minyawt, maxyawt, yawtw = minyaw, maxyaw, yaww

    P = Para(minx, miny, minyaw, minyawt, maxx, maxy, maxyaw,
             maxyawt, xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree)

    return P


def extract_path(closed, ngoal, nstart):
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


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or\
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def main():
    print("start!")

    sx = 14.0  # [m]
    sy = 10.0  # [m]
    syaw0 = np.deg2rad(00.0)
    syawt = np.deg2rad(00.0)

    gx = 0.0  # [m]
    gy = 0.0  # [m]
    gyaw0 = np.deg2rad(60.0)
    gyawt = np.deg2rad(60.0)

    ox, oy = [], []

    for i in range(-20, 20):
        ox.append(float(i))
        oy.append(15.0)

    for j in range(0, 5):
        ox.append(-10.0)
        oy.append(float(j))

    for j in range(0, 5):
        ox.append(10.0)
        oy.append(float(j))

    # for i in range(-10, 4):
    #     ox.append(-4.0)
    #     oy.append(float(i))

    path = hybrid_astar_planning(sx, sy, syaw0, syawt, gx, gy, gyaw0, gyawt,
                                 ox, oy, C.XY_RESO, C.YAW_RESO)

    plt.plot(ox, oy, ".k")
    lib.plot_trailer(sx, sy, syaw0, syawt, 0.0)
    lib.plot_trailer(gx, gy, gyaw0, gyawt, 0.0)
    x = path.x
    y = path.y
    yaw = path.yaw
    yawt = path.yawt
    direction = path.direction

    steer = 0.0
    for ii in range(len(x)):
        plt.cla()
        plt.plot(ox, oy, "sk")
        plt.plot(x, y, "-r", label="Hybrid A* path")

        if ii < len(x) - 2:
            k = (yaw[ii + 1] - yaw[ii]) / C.Motion_RESO
            if ~direction[ii]:
                k *= -1
            steer = math.atan2(C.WB * k, 1.0)
        else:
            steer = 0.0
        lib.plot_trailer(gx, gy, gyaw0, gyawt, 0.0)
        lib.plot_trailer(x[ii], y[ii], yaw[ii], yawt[ii], steer)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

    print("Done")
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
