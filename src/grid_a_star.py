"""
A* calc_astar_path
author: huiming zhou
"""

import heapq
import math
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind


def calc_dist_policy(gx, gy, ox, oy, reso, vr):
    """
    :param gx: goal x position [m]
    :param gy: goal y position [mm
    :param ox: x position list of obstacles [m]
    :param oy: y position list of obstacles [m]
    :param reso: grid resolution [m]
    :param vr: vehicle radius [m]
    """

    ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    ox = [iox / reso for iox in ox]
    oy = [ioy / reso for ioy in oy]

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, vr)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(ngoal, xw, minx, miny)] = ngoal

    motion = get_motion_model()
    nmotion = len(motion[:, 1])

    pq = []
    heapq.heappush(pq, (ngoal.cost, calc_index(ngoal, xw, minx, miny)))

    while True:
        if not open_set:
            break
        cost, c_id = heapq.heappop(pq)
        if c_id in open_set:
            current = open_set[c_id]
            closed_set[c_id] = current
            open_set.pop(c_id)
        else:
            continue

        for i, _ in enumerate(motion):
            node = Node(current.x + motion[i][0],
                        current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)

            if not verify_node(node, minx, miny, xw, yw, obmap):
                continue

            node_ind = calc_index(node, xw, minx, miny)

            if node_ind in closed_set:
                continue

            if node_ind in open_set:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind].cost = node.cost
                    open_set[node_ind].pind = c_id
            else:
                open_set[node_ind] = node
                heapq.heappush(pq, (node.cost, calc_index(node, xw, minx, miny)))

    pmap = calc_policy_map(closed_set, xw, yw, minx, miny)

    return pmap


def calc_policy_map(closed_set, xw, yw, minx, miny):
    pmap = [[float("inf") for _ in range(yw)] for _ in range(xw)]

    for n in closed_set.values():
        pmap[n.x - minx][n.y - miny] = n.cost

    return pmap


def calc_astar_path(sx, sy, gx, gy, ox, oy, reso, vr):
    """
    :param sx: start x position [m]
    :param sy: start y position [m]
    :param gx: goal x position [m]
    :param gy: goal y position [m]
    :param ox: x position list of Obstacles [m]
    :param oy: y position list of Obstacles [m]
    :param reso: grid resolution [m]
    :param vr: ridius of vehicle robot
    """

    nstart = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    ox = [iox / reso for iox in ox]
    oy = [ioy / reso for ioy in oy]

    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, vr)
    motion = get_motion_model()

    open_set, closed_set = dict(), dict()
    open_set[calc_index(nstart, xw, minx, miny)] = nstart
    pq = []
    heapq.heappush(pq, (calc_cost(nstart, ngoal), calc_index(nstart, xw, minx, miny)))

    while True:
        if not open_set:
            break

        cost, c_id = heapq.heappop(pq)
        current = open_set[c_id]

        if current.x == ngoal.x and current.y == ngoal.y:
            closed_set[c_id] = current

        open_set.pop(c_id)
        closed_set[c_id] = current

        for i, _ in enumerate(motion):
            node = Node(current.x + motion[i][0],
                        current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)

            if not verify_node(node, minx, miny, maxx, maxy, obmap):
                continue

            node_ind = calc_index(node, xw, minx, miny)

            if node_ind in closed_set:
                continue

            if node_ind in open_set:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind].cost = node.cost
                    open_set[node_ind].pind = c_id
            else:
                open_set[node_ind] = node
                heapq.heappush(pq, (calc_cost(node, ngoal), calc_index(node, xw, minx, miny)))

    rx, ry = get_final_path(closed_set, ngoal, nstart, xw, minx, miny, reso)

    return rx, ry


def verify_node(node, minx, miny, xw, yw, obmap):
    if node.x - minx >= xw:
        return False
    elif node.x - minx <= 0:
        return False
    if node.y - miny >= yw:
        return False
    elif node.y - miny <= 0:
        return False

    if obmap[node.x - minx][node.y - miny]:
        return False

    return True


def calc_cost(n, ngoal):
    return n.cost + h(n.x - ngoal.x, n.y - ngoal.y)


def get_motion_model():
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion


def calc_index(node, xwidth, xmin, ymin):
    return (node.y - ymin) * xwidth + (node.x - xmin)


def calc_obstacle_map(ox, oy, reso, vr):
    minx = round(min(ox))
    miny = round(min(oy))
    maxx = round(max(ox))
    maxy = round(max(oy))

    xwidth = round(maxx - minx)
    ywidth = round(maxy - miny)

    obmap = [[False for _ in range(ywidth)] for _ in range(xwidth)]

    for ix in range(xwidth):
        x = ix + minx
        for iy in range(ywidth):
            y = iy + miny
            for iox, ioy in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if d <= vr / reso:
                    obmap[ix][iy] = True
                    break

    return obmap, minx, miny, maxx, maxy, xwidth, ywidth


def get_final_path(closed_set, ngoal, nstart, xw, minx, miny, reso):
    rx, ry = [ngoal.x], [ngoal.y]
    nid = calc_index(ngoal, xw, minx, miny)

    while True:
        n = closed_set[nid]
        rx.append(n.x)
        ry.append(n.y)
        nid = n.pind

        if rx[-1] == nstart.x and ry[-1] == nstart.y:
            break

    rx = list(reversed(rx))
    ry = list(reversed(ry))

    rx = [x * reso for x in rx]
    ry = [y * reso for y in ry]

    return rx, ry


def search_min_cost_node(open_set, ngoal):
    mnode = None
    mcost = float("inf")

    for n in open_set.values():
        cost = n.cost + h(n.x - ngoal.x, n.y - ngoal.y)
        if mcost > cost:
            mnode = n
            mcost = cost

    return mnode


def h(x, y):
    return math.hypot(x, y)


def main():
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    VEHICLE_RADIUS = 1.0  # [m]
    GRID_RESOLUTION = 1.0  # [m]

    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    rx, ry = calc_astar_path(sx, sy, gx, gy, ox, oy, GRID_RESOLUTION, VEHICLE_RADIUS)

    plt.plot(ox, oy, "sk", label="obstacles")
    plt.plot(sx, sy, "xr", label="start")
    plt.plot(gx, gy, "xb", label="goal")
    plt.plot(rx, ry, "-r", label="A* path")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
