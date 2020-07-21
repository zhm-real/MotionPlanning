import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import Control.reeds_shepp_path as rs


class C:
    Kp = 1.0
    KTH = 1.0
    KE = 0.5
    dt = 0.1
    dref = 0.2

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width
    MAX_STEER = 0.25


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        self.x += self.v * math.cos(self.yaw) * C.dt
        self.y += self.v * math.sin(self.yaw) * C.dt
        self.yaw += self.v / C.WB * math.tan(delta) * C.dt
        self.direct = direct
        self.v += self.direct * a * C.dt


class Trajectory:
    def __init__(self, cx, cy, cyaw, ccurv):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ccurv = ccurv
        self.len = len(self.cx)
        self.s0 = 1

    def calc_track_error(self, node):
        s = self.nearest_point(node)

        k = self.ccurv[s]
        yaw = self.cyaw[s]

        dxl = node.x - self.cx[s]
        dyl = node.y - self.cy[s]
        angle = pi_2_pi(math.atan2(dyl, dxl) - yaw)
        e = math.hypot(dxl, dyl) * math.sin(abs(angle))

        if angle < 0:
            e *= -1

        return e, k, yaw, s

    def nearest_point(self, node):
        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        dist = np.hypot(dx, dy)
        self.s0 += np.argmin(dist[self.s0:self.len])

        return self.s0


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def pid_control(target_v, v, dist, direct):
    a = 0.2 * (target_v - direct * v)
    if dist > 10.0:
        a = 0.3 * (target_v - direct * v)
    else:
        if v > 2:
            a = -3.0
        elif v < -2:
            a = -1.0
    return a


def generate_path(s):
    max_c = math.tan(C.MAX_STEER) / C.WB
    path_x, path_y, yaw, direct, rc = [], [], [], [], []
    x_rec, y_rec, yaw_rec, direct_rec, rc_rec = [], [], [], [], []
    direc_flag = 1.0

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
            if idirect[j] == direc_flag:
                x_rec.append(ix[j])
                y_rec.append(iy[j])
                yaw_rec.append(iyaw[j])
                direct_rec.append(idirect[j])
                rc_rec.append(irc[j])
            else:
                if len(x_rec) == 0 or direct_rec[0] != direc_flag:
                    direc_flag = idirect[j]
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


def rear_wheel_feedback_control(state, e, k, yaw_ref):
    v = state.v
    th_e = pi_2_pi(state.yaw - yaw_ref)

    omega = v * k * math.cos(th_e) / (1.0 - k * e) - \
            C.KTH * abs(v) * th_e - C.KE * v * math.sin(th_e) * e / th_e

    if th_e == 0.0 or omega == 0.0:
        return 0.0

    delta = math.atan2(C.WB * omega / v, 1.0)

    return delta


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
        node = Node(x=x0, y=y0, yaw=yaw0, v=0.0, direct=cdirect[0])
        ref_trajectory = Trajectory(cx, cy, cyaw, ccurv)
        speed_ref = 30 / 3.6

        while t < maxTime:
            if cdirect[0] > 0:
                speed_ref = 30.0 / 3.6
                C.Ld = 3.5
            else:
                speed_ref = 20.0 / 3.6
                C.Ld = 2.5

            e, k, yawref, s0 = ref_trajectory.calc_track_error(node)
            di = rear_wheel_feedback_control(node, e, k, yawref)

            dist = math.hypot(node.x - cx[-1], node.y - cy[-1])

            ai = pid_control(speed_ref, node.v, dist, node.direct)
            node.update(ai, di, node.direct)
            t += C.dt

            if dist <= C.dref:
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
            plt.plot(cx[s0], cy[s0], '.r')
            draw.draw_car(node.x, node.y, node.yaw, steer, C)
            plt.axis("equal")
            plt.title("RearWheelFeedback: v=" + str(node.v * 3.6)[:4] + "km/h")
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    main()
