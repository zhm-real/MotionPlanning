"""
LQR and PID Controller
author: huiming zhou
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs


class C:
    # PID config
    Kp = 1.0

    # System config
    dt = 0.1
    dist_stop = 0.5
    Q = np.eye(4)
    R = np.eye(1)

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width
    MAX_STEER = 0.30


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        # delta = self.limit_input(delta)
        self.x += self.v * math.cos(self.yaw) * C.dt
        self.y += self.v * math.sin(self.yaw) * C.dt
        self.yaw += self.v / C.WB * math.tan(delta) * C.dt
        self.direct = direct
        self.v += self.direct * a * C.dt

    @staticmethod
    def limit_input(delta):
        if delta >= C.MAX_STEER:
            return C.MAX_STEER

        if delta <= -C.MAX_STEER:
            return -C.MAX_STEER

        return delta


class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.ind_old = 0
        self.len = len(self.cx)

    def calc_theta_e_and_er(self, node):
        """
        calc theta_e and er.
        theta_e = yaw_car - yaw_ref_path
        er = lateral distance in frenet frame

        :param node: current information of vehicle
        :return: theta_e and er
        """

        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        ind = int(np.argmin(np.hypot(dx, dy)))

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind]],
                                      [dy[ind]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        theta = node.yaw
        theta_p = self.cyaw[ind]
        theta_e = pi_2_pi(theta - theta_p)

        k = self.ck[ind]

        return theta_e, er, k, ind


def lqr_lateral_control(node, er_old, theta_e_old, ref_path):
    """
    using lqr controller calc optimal steering angle of vehicle.
    model:
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    state vector:
        x[k] = [er[k] (er[k] - er[k-1])/dt theta_e[k] (theta_e[k] - theta_e[k-1])/dt].T
    input vector:
        u[k] = [delta[k]]

    :param node: current information of vehicle
    :param er_old: lateral position of last time
    :param theta_e_old: theta error of last time
    :param ref_path: reference path: x, y, yaw, curvature
    :return: optimal input delta_optimal = delta_feedback + delta_feedforward
    """

    theta_e, er, k, ind = ref_path.calc_theta_e_and_er(node)
    A, B = calc_linearized_discrete_model(node.v)
    Q = C.Q
    R = C.R
    P = solve_riccati_equation(A, B, Q, R)
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    X = np.array([[er],
                  [(er - er_old) / C.dt],
                  [theta_e],
                  [(theta_e - theta_e_old) / C.dt]])
    delta_fb = -K @ X
    delta_ff = C.WB * k
    delta = delta_ff + delta_fb

    return delta, er, theta_e, ind


def solve_riccati_equation(A, B, Q, R):
    """
    solve a discrete-time algebraic riccati equation.
    Course Optimal Control verified that value iteration could solve raccati equation.
    :param A: state-space equation A
    :param B: state-space equation B
    :param Q: penalty on state vector x
    :param R: penalty on input vector u
    :return: P: d(x.TPx)/dt = -x.T(Q+K.TRK)x
    """

    P = Q
    P_next = Q
    iter_max = 100
    eps = 0.01

    for k in range(iter_max):
        P_next = A.T @ P @ A - A.T @ P @ B @ \
                 np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q

        if (abs(P_next - P)).max() < eps:
            break

        P = P_next

    return P_next


def calc_linearized_discrete_model(v):
    """
    kinematic model -> linear system -> discrete-time linear system
    :param v: current velocity
    :return: state-space equation: A and B
    """

    A = np.array([[1.0, C.dt, 0.0, 0.0],
                  [0.0, 0.0, v, 0.0],
                  [0.0, 0.0, 1.0, C.dt],
                  [0.0, 0.0, 0.0, 0.0]])

    B = np.array([[0.0],
                  [0.0],
                  [0.0],
                  [v / C.WB]])

    return A, B


def pid_longitudinal_control(target_v, v, dist, direct):
    """
    using LQR as lateral controller, PID as longitudinal controller (speed control)
    :param target_v: target speed
    :param v: current speed
    :param dist: distance to end point
    :param direct: current direction of vehicle, 1.0: forward, -1.0: backward
    :return: acceleration
    """

    a = 0.3 * (target_v - direct * v)

    if dist < 10.0:
        if v > 2:
            a = -3.0
        elif v < -2:
            a = -1.0

    return a


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def generate_path(s):
    """
    design path using reeds-shepp path generator.
    divide paths into sections, in each section the direction is the same.
    :param s: objective positions and directions.
    :return: paths
    """

    max_c = math.tan(C.MAX_STEER) / C.WB
    path_x, path_y, yaw, direct, rc = [], [], [], [], []
    x_rec, y_rec, yaw_rec, direct_rec, rc_rec = [], [], [], [], []
    direct_flag = 1.0

    for i in range(len(s) - 1):
        s_x, s_y, s_yaw = s[i][0], s[i][1], np.deg2rad(s[i][2])
        g_x, g_y, g_yaw = s[i + 1][0], s[i + 1][1], np.deg2rad(s[i + 1][2])

        path_i = rs.calc_optimal_path(s_x, s_y, s_yaw,
                                      g_x, g_y, g_yaw, max_c)

        irc, rds = rs.calc_curvature(path_i.x, path_i.y, path_i.yaw, path_i.directions)

        ix = path_i.x
        iy = path_i.y
        iyaw = path_i.yaw
        idirect = path_i.directions

        for j in range(len(ix)):
            if idirect[j] == direct_flag:
                x_rec.append(ix[j])
                y_rec.append(iy[j])
                yaw_rec.append(iyaw[j])
                direct_rec.append(idirect[j])
                rc_rec.append(irc[j])
            else:
                if len(x_rec) == 0 or direct_rec[0] != direct_flag:
                    direct_flag = idirect[j]
                    continue

                path_x.append(x_rec)
                path_y.append(y_rec)
                yaw.append(yaw_rec)
                direct.append(direct_rec)
                rc.append(rc_rec)
                x_rec, y_rec, yaw_rec, direct_rec, rc_rec = \
                    [x_rec[-1]], [y_rec[-1]], [yaw_rec[-1]], [-direct_rec[-1]], [rc_rec[-1]]

    path_x.append(x_rec)
    path_y.append(y_rec)
    yaw.append(yaw_rec)
    direct.append(direct_rec)
    rc.append(rc_rec)

    x_all, y_all = [], []
    for ix, iy in zip(path_x, path_y):
        x_all += ix
        y_all += iy

    return path_x, path_y, yaw, direct, rc, x_all, y_all


def main():
    # generate path
    states = [(0, 0, 0), (20, 15, 0), (35, 20, 90), (40, 0, 180),
              (20, 0, 120), (5, -10, 180), (15, 5, 30)]
    #
    # states = [(-3, 3, 120), (10, -7, 30), (10, 13, 30), (20, 5, -25),
    #           (35, 10, 180), (30, -10, 160), (5, -12, 90)]

    x_ref, y_ref, yaw_ref, direct, curv, x_all, y_all = generate_path(states)

    maxTime = 100.0
    yaw_old = 0.0
    x0, y0, yaw0, direct0 = \
        x_ref[0][0], y_ref[0][0], yaw_ref[0][0], direct[0][0]

    x_rec, y_rec, yaw_rec, direct_rec = [], [], [], []

    for cx, cy, cyaw, cdirect, ccurv in zip(x_ref, y_ref, yaw_ref, direct, curv):
        t = 0.0
        er, theta_e = 0.0, 0.0

        node = Node(x=x0, y=y0, yaw=yaw0, v=0.0, direct=cdirect[0])
        ref_path = PATH(cx, cy, cyaw, ccurv)

        while t < maxTime:
            if cdirect[0] > 0:
                speed_ref = 30.0 / 3.6
            else:
                speed_ref = 20.0 / 3.6

            delta, er, theta_e, ind = lqr_lateral_control(node, er, theta_e, ref_path)

            dist = math.hypot(node.x - cx[-1], node.y - cy[-1])

            acceleration = pid_longitudinal_control(speed_ref, node.v, dist, node.direct)
            node.update(acceleration, delta, node.direct)
            t += C.dt

            if dist <= C.dist_stop:
                break

            x_rec.append(node.x)
            y_rec.append(node.y)
            yaw_rec.append(node.yaw)
            direct_rec.append(node.direct)

            dy = (node.yaw - yaw_old) / (node.v * C.dt)
            steer = rs.pi_2_pi(-math.atan(C.WB * dy))

            yaw_old = node.yaw
            x0 = x_rec[-1]
            y0 = y_rec[-1]
            yaw0 = yaw_rec[-1]

            plt.cla()
            plt.plot(x_all, y_all, color='gray', linewidth=2.0)
            plt.plot(x_rec, y_rec, linewidth=2.0, color='darkviolet')
            plt.plot(cx[ind], cy[ind], '.r')
            draw.draw_car(node.x, node.y, node.yaw, steer, C)
            plt.axis("equal")
            plt.title("LQR & PID: v=" + str(node.v * 3.6)[:4] + "km/h")
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event:
                                         [exit(0) if event.key == 'escape' else None])
            plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    main()
