import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

import astar
import reeds_shepp_path as rs


class C:  # Parameter config
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

    LF = 4.5
    LB = 1.0
    W = 2.6
    WB = 3.7  # [m] Wheel base
    TR = 0.5  # Tyre radius [m] for plot
    TW = 1.0  # Tyre width [m] for plot
    MAX_STEER = 0.6  # [rad] maximum steering angle


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


def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
    syaw, gyaw = rs.pi_2_pi(syaw), rs.pi_2_pi(gyaw)

    nstart = Node(round(sx / xyreso), round(sy / xyreso), round(syaw / yawreso),
                  1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(round(gx / xyreso), round(gy / xyreso), round(gyaw / yawreso),
                 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)

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

        isupdated, fpath = update_node_with_analystic_expantion(n_curr, ngoal, gyaw, P)

        if isupdated:
            fnode = fpath
            break

        for i in range(len(steer)):
            node = calc_next_node(n_curr, ind, steer[i], direc[i], P)

            if not verify_index(node, P):
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


def extract_path(closed, ngoal, nstart):
    rx = list(reversed(ngoal.x))
    ry = list(reversed(ngoal.y))
    ryaw = list(reversed(ngoal.yaw))
    direction = list(reversed(ngoal.directions))
    nid = ngoal.pind
    finalcost = ngoal.cost

    while True:
        n = closed[nid]
        rx = rx + list(reversed(n.x))
        ry = ry + list(reversed(n.y))
        ryaw = ryaw + list(reversed(n.yaw))
        direction = direction + list(reversed(n.directions))
        nid = n.pind

        if is_same_grid(n, nstart):
            break

    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    direction = list(reversed(direction))

    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, direction, finalcost)

    return path


def verify_index(node, P):
    if node.xind <= P.minx or \
            node.xind >= P.maxx or \
            node.yind <= P.miny or \
            node.yind >= P.maxy:
        return False

    return True


def calc_next_node(n_curr, c_id, u, d, c):
    arc_l = C.XY_RESO * 1.5

    nlist = int(arc_l / C.Motion_RESO) + 1
    xlist = [0.0 for _ in range(nlist)]
    ylist = [0.0 for _ in range(nlist)]
    yawlist = [0.0 for _ in range(nlist)]

    xlist[0] = n_curr.x[-1] + d * C.Motion_RESO * math.cos(n_curr.yaw[-1])
    ylist[0] = n_curr.y[-1] + d * C.Motion_RESO * math.sin(n_curr.yaw[-1])
    yawlist[0] = rs.pi_2_pi(n_curr.yaw[-1] + d * C.Motion_RESO / C.WB * math.tan(u))

    for i in range(nlist - 1):
        xlist[i + 1] = xlist[i] + d * C.Motion_RESO * math.cos(yawlist[i])
        ylist[i + 1] = ylist[i] + d * C.Motion_RESO * math.sin(yawlist[i])
        yawlist[i + 1] = rs.pi_2_pi(yawlist[i] + d * C.Motion_RESO / C.WB * math.tan(u))

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

    cost = n_curr.cost + addedcost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, directions, u, cost, c_id)

    return node


def update_node_with_analystic_expantion(n_curr, ngoal, gyaw, P):
    apath = analystic_expantion(n_curr, ngoal, P)  # rs path: n_curr -> ngoal

    if apath:
        fx = apath.x[1:-1]
        fy = apath.y[1:-1]
        fyaw = apath.yaw[1:-1]

        if abs(rs.pi_2_pi(n_curr.yaw[-1] - gyaw)) >= C.GOAL_TYAW_TH:
            return False, None

        fcost = n_curr.cost + calc_rs_path_cost(apath)
        fpind = calc_index(n_curr, P)

        fsteer = 0.0
        fd = apath.directions[1:-1]
        fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                     fx, fy, fyaw, fd, fsteer, fcost, fpind)

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
        heapq.heappush(pathqueue, (calc_rs_path_cost(path), path))

    _, path = heapq.heappop(pathqueue)
    ind = range(0, len(path.x), C.SKIP_COLLISION_CHECK)

    pathx = [path.x[k] for k in ind]
    pathy = [path.y[k] for k in ind]
    pathyaw = [path.yaw[k] for k in ind]

    if check_collision(pathx, pathy, pathyaw, P):
        return path

    return None


def check_collision(x, y, yaw, P):
    vrx = [C.LF, C.LF, -C.LB, -C.LB, C.LF]
    vry = [-C.W / 2.0, C.W / 2.0, C.W / 2.0, -C.W / 2.0, -C.W / 2.0]

    wbd = (C.LF + C.LB) / 2.0 - C.LB
    wbr = (C.LF + C.LB) / 2.0 + 0.3

    for ix, iy, iyaw in zip(x, y, yaw):
        cx = ix + wbd * math.cos(iyaw)
        cy = iy + wbd * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], wbr)

        if not ids:
            continue

        obsx = [P.ox[i] for i in ids]
        obsy = [P.oy[i] for i in ids]

        if not check_rectangle(ix, iy, iyaw, obsx, obsy, vrx, vry):
            return False

    return True


def check_rectangle(ix, iy, iyaw, ox, oy, vrx, vry):
    c = math.cos(-iyaw)
    s = math.sin(-iyaw)

    for iox, ioy in zip(ox, oy):
        tx = iox - ix
        ty = ioy - iy
        lx = (c * tx - s * ty)
        ly = (s * tx + c * ty)

        sumangle = 0.0

        for i in range(len(vrx) - 1):
            x1 = vrx[i] - lx
            y1 = vry[i] - ly
            x2 = vrx[i + 1] - lx
            y2 = vry[i + 1] - ly
            d1 = math.hypot(x1, y1)
            d2 = math.hypot(x2, y2)
            theta1 = math.atan2(y1, x1)
            tty = (-math.sin(theta1) * x2 + math.cos(theta1) * y2)
            tmp = (x1 * x2 + y1 * y2) / (d1 * d2)

            if tmp >= 1.0:
                tmp = 1.0
            elif tmp <= 0.0:
                tmp = 0.0

            if tty >= 0.0:
                sumangle += math.acos(tmp)
            else:
                sumangle -= math.acos(tmp)

        if sumangle >= C.PI:
            return False

    return True


def calc_rs_path_cost(rspath):
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

    return cost


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def get_motion():
    s = [i for i in np.arange(C.MAX_STEER / C.N_STEER,
                              C.MAX_STEER, C.MAX_STEER / C.N_STEER)]

    steer = [0.0] + s + [-i for i in s]
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    return ind


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

    P = Para(minx, miny, minyaw, maxx, maxy, maxyaw,
             xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)

    return P


def plot_trailer(x, y, yaw, steer):
    truckcolor = "-k"

    LENGTH = C.LB + C.LF

    truckOutLine = np.array([[-C.LB, (LENGTH - C.LB), (LENGTH - C.LB), (-C.LB), (-C.LB)],
                             [C.W / 2, C.W / 2, -C.W / 2, -C.W / 2, C.W / 2]])

    rr_wheel = np.array([[C.TR, -C.TR, -C.TR, C.TR, C.TR],
                         [-C.W / 12.0 + C.TW, -C.W / 12.0 + C.TW, C.W / 12.0 + C.TW,
                          C.W / 12.0 + C.TW, -C.W / 12.0 + C.TW]])

    rl_wheel = np.array([[C.TR, -C.TR, -C.TR, C.TR, C.TR],
                         [-C.W / 12.0 - C.TW, -C.W / 12.0 - C.TW, C.W / 12.0 - C.TW,
                          C.W / 12.0 - C.TW, -C.W / 12.0 - C.TW]])

    fr_wheel = np.array([[C.TR, -C.TR, -C.TR, C.TR, C.TR],
                         [-C.W / 12.0 + C.TW, -C.W / 12.0 + C.TW, C.W / 12.0 + C.TW,
                          C.W / 12.0 + C.TW, -C.W / 12.0 + C.TW]])

    fl_wheel = np.array([[C.TR, -C.TR, -C.TR, C.TR, C.TR],
                         [-C.W / 12.0 - C.TW, -C.W / 12.0 - C.TW, C.W / 12.0 - C.TW,
                          C.W / 12.0 - C.TW, -C.W / 12.0 - C.TW]])

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(-steer), math.sin(-steer)],
                     [-math.sin(-steer), math.cos(-steer)]])

    fr_wheel = (np.dot(fr_wheel.T, Rot2)).T
    fl_wheel = (np.dot(fl_wheel.T, Rot2)).T

    fr_wheel[0, :] = fr_wheel[0, :] + C.WB
    fl_wheel[0, :] = fl_wheel[0, :] + C.WB

    fr_wheel = (np.dot(fr_wheel.T, Rot1)).T
    fl_wheel = (np.dot(fl_wheel.T, Rot1)).T

    truckOutLine = (np.dot(truckOutLine.T, Rot1)).T
    rr_wheel = (np.dot(rr_wheel.T, Rot1)).T
    rl_wheel = (np.dot(rl_wheel.T, Rot1)).T

    truckOutLine[0, :] += x
    truckOutLine[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(truckOutLine[0, :], truckOutLine[1, :], truckcolor)
    plt.plot(fr_wheel[0, :], fr_wheel[1, :], truckcolor)
    plt.plot(rr_wheel[0, :], rr_wheel[1, :], truckcolor)
    plt.plot(fl_wheel[0, :], fl_wheel[1, :], truckcolor)
    plt.plot(rl_wheel[0, :], rl_wheel[1, :], truckcolor)
    plt.plot(x, y, "*")


def main():
    print("start!")

    sx = 0.0  # [m]
    sy = 10.0  # [m]
    syaw0 = np.deg2rad(00.0)

    gx = 0.0  # [m]
    gy = 3.0  # [m]
    gyaw0 = np.deg2rad(180.0)

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

    #
    # for j in range(0, 5):
    #     ox.append(10.0)
    #     oy.append(float(j))

    # for i in range(-10, 4):
    #     ox.append(-4.0)
    #     oy.append(float(i))

    path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,
                                 ox, oy, C.XY_RESO, C.YAW_RESO)

    plt.plot(ox, oy, ".k")
    plot_trailer(sx, sy, syaw0, 0.0)
    plot_trailer(gx, gy, gyaw0, 0.0)
    x = path.x
    y = path.y
    yaw = path.yaw
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
        plot_trailer(gx, gy, gyaw0, 0.0)
        plot_trailer(x[ii], y[ii], yaw[ii], steer)
        plt.grid(True)
        plt.axis("equal")
        plt.pause(0.0001)

    print("Done")
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
