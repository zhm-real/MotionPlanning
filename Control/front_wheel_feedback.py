import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs


class C:
    Kp = 1.0
    k = 0.5
    dt = 0.1
    dref = 0.5
    max_steer = np.deg2rad(30.0)

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
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * C.dt
        self.y += self.v * math.sin(self.yaw) * C.dt
        self.yaw += self.v / C.WB * math.tan(delta) * C.dt
        self.v += a * C.dt


class Trajectory:
    def __init__(self, cx, cy, cyaw):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ind_old = 0

    def target_index(self, node):
        fx = node.x + C.WB * math.cos(node.yaw)
        fy = node.y + C.WB * math.sin(node.yaw)

        dx = [fx - x for x in self.cx]
        dy = [fy - y for y in self.cy]
        target_index = np.argmin(np.hypot(dx, dy))

        front_axle_vec = [-np.cos(node.yaw + np.pi / 2),
                          -np.sin(node.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_index], dy[target_index]], front_axle_vec)

        return target_index, error_front_axle

    def front_wheel_feedback_control(self, node):
        target_index, ef = self.target_index(node)
        if self.ind_old >= target_index:
            target_index = self.ind_old
        else:
            self.ind_old = target_index

        theta_e = pi_2_pi(self.cyaw[target_index] - node.yaw)
        theta_d = math.atan2(C.k * ef, node.v)
        delta = theta_e + theta_d

        return delta, target_index


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def pid_control(target_v, v, dist):
    a = 0.2 * (target_v - v)
    if dist > 10.0:
        a = 0.3 * (target_v - v)
    else:
        if v > 2:
            a = -3.0
        elif v < -2:
            a = -1.0
    return a


def main():
    # generate path
    ax = np.arange(0, 50, 0.5)
    ay = [math.sin(ix / 5.0) * ix / 2.0 for ix in ax]

    cx, cy, cyaw, _, _ = cs.calc_spline_course(ax, ay, ds=C.dt)

    t = 0.0
    maxTime = 100.0
    yaw_old = 0.0
    x0, y0, yaw0 = cx[0], cy[0], cyaw[0]
    xrec, yrec, yawrec = [], [], []

    node = Node(x=x0, y=y0, yaw=yaw0, v=0.0)
    ref_trajectory = Trajectory(cx, cy, cyaw)

    while t < maxTime:
        speed_ref = 25.0 / 3.6
        C.Ld = 3.5

        di, target_index = ref_trajectory.front_wheel_feedback_control(node)

        dist = math.hypot(node.x - cx[-1], node.y - cy[-1])
        ai = pid_control(speed_ref, node.v, dist)
        node.update(ai, di)
        t += C.dt

        if dist <= C.dref:
            break

        dy = (node.yaw - yaw_old) / (node.v * C.dt)
        steer = rs.pi_2_pi(-math.atan(C.WB * dy))
        yaw_old = node.yaw

        xrec.append(node.x)
        yrec.append(node.y)
        yawrec.append(node.yaw)

        plt.cla()
        plt.plot(cx, cy, color='gray', linewidth=2.0)
        plt.plot(xrec, yrec, linewidth=2.0, color='darkviolet')
        plt.plot(cx[target_index], cy[target_index], '.r')
        draw.draw_car(node.x, node.y, node.yaw, steer, C)
        plt.axis("equal")
        plt.title("FrontWheelFeedback: v=" + str(node.v * 3.6)[:4] + "km/h")
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    main()


