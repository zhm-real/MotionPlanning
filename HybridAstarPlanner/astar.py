import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node


class Para:
    """
    整体寻路的配置 存储了整个寻路的参数
    """
    def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.xw = xw # width of grid world
        self.yw = yw # height of grid world
        self.reso = reso  # resolution of grid world
        self.motion = motion  # motion set


def astar_planning(start_x, start_y, goal_x, goal_y, obstacle_x_list, obstacle_y_list, reso, rr):
    """
    return path of A*.
    :param start_x: starting node x [m]  # 起始节点的 x 坐标（单位：米）
    :param start_y: starting node y [m]  # 起始节点的 y 坐标（单位：米）
    :param goal_x: goal node x [m]  # 目标节点的 x 坐标（单位：米）
    :param goal_y: goal node y [m]  # 目标节点的 y 坐标（单位：米）
    :param obstacle_x_list: obstacles x positions [m]  # 障碍物的 x 坐标列表（单位：米）
    :param obstacle_y_list: obstacles y positions [m]  # 障碍物的 y 坐标列表（单位：米）
    :param reso: xy grid resolution  # 网格分辨率
    :param rr: robot radius  # 机器人半径
    :return: path
    """

    n_start = Node(round(start_x / reso), round(start_y / reso), 0.0, -1)  # 创建起始节点
    n_goal = Node(round(goal_x / reso), round(goal_y / reso), 0.0, -1)  # 创建目标节点

    obstacle_x_list = [x / reso for x in obstacle_x_list]  # 将障碍物 x 坐标进行网格化处理
    obstacle_y_list = [y / reso for y in obstacle_y_list]  # 将障碍物 y 坐标进行网格化处理

    P, obsmap = calc_parameters(obstacle_x_list, obstacle_y_list, rr, reso)  # 计算路径规划所需的参数

    open_set, closed_set = dict(), dict()  # 创建开放集和封闭集的字典
    open_set[calc_index(n_start, P)] = n_start  # 将起始节点加入开放集

    q_priority = []  # 创建优先队列
    heapq.heappush(q_priority, (fvalue(n_start, n_goal) + 0, calc_index(n_start, P)))  # 将起始节点的优先级和索引加入优先队列

    while True:  # 主循环
        if not open_set:  # 如果开放集为空，则退出循环
            break

        _, ind = heapq.heappop(q_priority)  # 从优先队列中取出优先级最高的节点
        n_curr = open_set[ind]  # 从开放集中获取当前节点
        closed_set[ind] = n_curr  # 将当前节点加入封闭集
        open_set.pop(ind)  # 从开放集中移除当前节点

        for i in range(len(P.motion)):  # 遍历节点的运动动作下的临近结点
            # (0) 初始化邻节点(包括其g value 和 parent index)
            node = Node(n_curr.x + P.motion[i][0], n_curr.y + P.motion[i][1], n_curr.cost + u_cost(P.motion[i]), ind)  # 生成新的节点

            # (1) 检查该邻节点的有效性(可探索性)
            if not check_node(node, P, obsmap):
                continue  # 如果节点无效，则继续下一个循环

            # (2) 按照邻结点的类别分别处理(新结点，已在open，已在closed)
            n_ind = calc_index(node, P)  # 计算新节点的索引
            if n_ind not in closed_set:  # 如果新节点不在封闭集中，代表是新节点(未被探索)
                if n_ind in open_set:  # 如果新节点在开放集中，代表是旧节点(已被探索)，需要更新代价
                    if open_set[n_ind].cost > node.cost:  # 如果新节点的代价更低
                        open_set[n_ind].cost = node.cost  # 更新开放集中该节点的代价
                        open_set[n_ind].pind = ind  # 更新开放集中该节点的父节点索引
                else:
                    open_set[n_ind] = node  # 如果新节点不在开放集中，将新节点加入开放集，以供后续探索
                    heapq.heappush(q_priority, (fvalue(node, n_goal), calc_index(node, P)))  # 将新节点的优先级和索引加入优先队列

    pathx, pathy = extract_path(closed_set, n_start, n_goal, P)  # 提取路径，按照父亲结点索引从终点往回找到起点

    return pathx, pathy  # 返回路径的 x 坐标列表和 y 坐标列表


def calc_holonomic_heuristic_with_obstacle(node, obstacle_x_list, obstacle_y_list, reso, rr):
    """
    这是一个功能函数，没有在astar模块中用到
    而是在HA*中使用到的
    :param node:
    :param obstacle_x_list:
    :param obstacle_y_list:
    :param reso:
    :param rr:
    :return:
    """
    n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)

    obstacle_x_list = [x / reso for x in obstacle_x_list]
    obstacle_y_list = [y / reso for y in obstacle_y_list]

    P, obsmap = calc_parameters(obstacle_x_list, obstacle_y_list, reso, rr)

    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal

    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))

    while True:
        if not open_set:
            break

        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)

            if not check_node(node, P, obsmap):
                continue

            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_index(node, P)))

    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]

    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost

    return hmap


def check_node(node, P, obsmap):
    """
    检查结点是否有效，既不能是障碍物，也不能是在地图外
    :param node:
    :param P:
    :param obsmap:
    :return:
    """
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy:
        # 如果节点超出地图范围，则返回 False
        return False

    if obsmap[node.x - P.minx][node.y - P.miny]:
        # 如果节点在障碍物内，则返回 False
        return False
    # 如果节点在地图范围内且不在障碍物内，则返回 True
    return True


def u_cost(u):
    """
    计算u向量的模长
    :param u:
    :return:
    """
    return math.hypot(u[0], u[1])


def fvalue(node, n_goal):
    """
    A*算法的代价函数，这里采用的是g(n) + h(n)
    :param node:
    :param n_goal:
    :return:
    """
    return node.cost + h(node, n_goal)


def h(node, n_goal):
    """
    A*算法的启发值heuristic，这里采用距离目标点的欧式距离
    :param node:
    :param n_goal:
    :return:
    """
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)

def calc_index(node, P)->int:
    """
    计算节点的唯一字典索引
    这里比较巧妙的是：
        unique_index = delta_y * xw + delta_x
        这样每一个结点都有一个唯一的索引值
    :param node:
    :param P:
    :return:
    """
    return (node.y - P.miny) * P.xw + (node.x - P.minx)


def calc_parameters(obstacle_x_list, obstacle_y_list, rr, reso):
    # 根据提供的障碍物列表 动态确定地图的大小
    minx, miny = round(min(obstacle_x_list)), round(min(obstacle_y_list))
    maxx, maxy = round(max(obstacle_x_list)), round(max(obstacle_y_list))
    xw, yw = maxx - minx, maxy - miny  # 地图的宽度和高度


    motion = get_motion()  # 获取运动方向模型
    paramters = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)  # 定义参数对象
    obsmap = calc_obsmap(obstacle_x_list, obstacle_y_list, rr, paramters)  # 计算障碍物地图

    return paramters, obsmap


def calc_obsmap(obstacle_x_list, obstacle_y_list, rr, P) -> List[List[bool]]:
    """
    通过配置参数P来初始化True和False的障碍物地图。
    True表示有障碍物 False表示没有障碍物
    返回一个(xw,yw)大小的二维列表
    :param obstacle_x_list:
    :param obstacle_y_list:
    :param rr:
    :param P:
    :return:
    """
    # 初始化障碍物地图
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]

    # 遍历地图中的每一个点，如果该点到障碍物的距离小于rr，则该点为障碍物
    for x in range(P.xw):
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for obstacle_x_listx, obstacle_y_listy in zip(obstacle_x_list, obstacle_y_list):
                # 如果该点到障碍物的距离小于rr，则该点为障碍物
                if math.hypot(obstacle_x_listx - xx, obstacle_y_listy - yy) <= rr / P.reso:
                    obsmap[x][y] = True
                    break

    return obsmap  # 返回一个(xw,yw)大小的二维列表


def extract_path(closed_set, n_start, n_goal, P):
    pathx, pathy = [n_goal.x], [n_goal.y]
    n_ind = calc_index(n_goal, P)

    while True:
        node = closed_set[n_ind]
        pathx.append(node.x)
        pathy.append(node.y)
        n_ind = node.pind

        if node == n_start:
            break

    pathx = [x * P.reso for x in reversed(pathx)]
    pathy = [y * P.reso for y in reversed(pathy)]

    return pathx, pathy


def get_motion():
    """
    生成机器人的运动模型，motion 为八个方向的运动模型，每个元素为一个二维向量。
    就是get child函数
    :return:
    """
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]

    return motion


def get_env() -> tuple[List[float], List[float]]:
    """
    生成环境地图中的障碍物，obstacle_x_list、obstacle_y_list 分别为障碍物的 x 和 y 坐标。
    障碍物的大小为 grid_resolution * grid_resolution。
    :return: 
    """
    obstacle_x_list, obstacle_y_list = [], []
    
    # 四周的环境墙壁
    for i in range(60):
        obstacle_x_list.append(i)
        obstacle_y_list.append(0.0)
    for i in range(60):
        obstacle_x_list.append(60.0)
        obstacle_y_list.append(i)
    for i in range(61):
        obstacle_x_list.append(i)
        obstacle_y_list.append(60.0)
    for i in range(61):
        obstacle_x_list.append(0.0)
        obstacle_y_list.append(i)
    # 中间的障碍物
    for i in range(40):
        obstacle_x_list.append(20.0)
        obstacle_y_list.append(i)
    for i in range(40):
        obstacle_x_list.append(40.0)
        obstacle_y_list.append(60.0 - i)

    return obstacle_x_list, obstacle_y_list


def main():
    start_x = 10.0  # [m] 起始点的 x 坐标，单位为米。
    start_y = 50.0  # [m] 起始点的 y 坐标，单位为米。
    goal_x = 50.0  # [m]
    goal_y = 50.0  # [m]

    robot_radius = 2.0  # 机器人的半径，单位为米。用于考虑机器人尺寸在路径规划中的碰撞避免。
    grid_resolution = 1.0  # 网格分辨率，单位为米。用于将环境离散化为网格地图，以进行路径规划。
    obstacle_x_list, obstacle_y_list = get_env()  # 得出障碍物

    pathx, pathy = astar_planning(start_x, start_y, goal_x, goal_y, obstacle_x_list, obstacle_y_list, grid_resolution, robot_radius)

    plt.plot(obstacle_x_list, obstacle_y_list, 'sk')
    plt.plot(pathx, pathy, '-r')
    plt.plot(start_x, start_y, 'sg')
    plt.plot(goal_x, goal_y, 'sb')
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
