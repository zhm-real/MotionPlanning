import math
import numpy as np
import matplotlib.pyplot as plt

import draw
import reeds_shepp_path as rs
import pycubicspline.pycubicspline as cs


class C:
    Kp = 1.0
    KTH = 1.0
    KE = 0.5
    dt = 0.1
    dref = 0.5
    lqr_Q = np.eye(5)
    lqr_R = np.eye(2)

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width
    MAX_STEER = 0.8


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        if delta >= C.MAX_STEER:
            delta = C.MAX_STEER
        if delta <= - C.MAX_STEER:
            delta = - C.MAX_STEER
        self.x += self.v * math.cos(self.yaw) * C.dt
        self.y += self.v * math.sin(self.yaw) * C.dt
        self.yaw += self.v / C.WB * math.tan(delta) * C.dt
        self.direct = direct
        self.v += self.direct * a * C.dt


class Trajectory:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.ind_old = 0
        self.len = len(self.cx)

    def lqr_control(self, node, pe, pth_e, sp, Q, R):
        ind, e = self.nearest_index(node)
        tv = sp[ind]
        k = self.ck[ind]
        v = node.v
        th_e = pi_2_pi(node.yaw - self.cyaw[ind])

        A = np.array([[1.0, C.dt, 0.0, 0.0, 0.0],
                      [0.0, 0.0, v, 0.0, 0.0],
                      [0.0, 0.0, 1.0, C.dt, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0]])

        B = np.array([[0.0, 0.0],
                      [0.0, 0.0],
                      [0.0, 0.0],
                      [v / C.WB, 0.0],
                      [0.0, C.dt]])

        K, _, _ = dlqr(A, B, Q, R)

        x = np.array([[e],
                      [(e - pe) / C.dt],
                      [th_e],
                      [(th_e - pth_e) / C.dt],
                      [v - tv]])

        u_optimal = -K @ x

        ff = math.atan2(C.WB * k, 1)  # feedforward steering angle
        fb = pi_2_pi(u_optimal[0, 0])  # feedback steering angle
        delta = (ff + fb)
        accel = u_optimal[1, 0]

        return delta, ind, e, th_e, accel

    def nearest_index(self, node):
        dx = [node.x - x for x in self.cx]
        dy = [node.y - y for y in self.cy]
        dist = np.hypot(dx, dy)
        ind = np.argmin(dist)
        dist_min = dist[ind]

        dxl = self.cx[ind] - node.x
        dyl = self.cy[ind] - node.y
        angle = pi_2_pi(self.cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            dist_min *= -1

        return ind, dist_min


def dlqr(A, B, Q, R):
    """
    Solve the discrete T lqr controller
    x[k+1] = A*x[k] + B*u[k]
    cost function = sum(x[k].T*Q*x[k] + u[k].T*R*u[k])
    """
    X = solve_ricatti(A, B, Q, R)
    K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
    eig_result = np.linalg.eig(A - B @ K)

    return K, X, eig_result[0]


def solve_ricatti(A, B, Q, R):
    x = Q
    x_next = Q
    max_iter = 150
    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 np.linalg.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next


def calc_speed_profile(cyaw, target_speed):
    speed_profile = [target_speed] * len(cyaw)

    direction = 1.0

    # Set stop point
    for i in range(len(cyaw) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    # speed down
    for i in range(40):
        speed_profile[-i] = target_speed / (50 - i)
        if speed_profile[-i] <= 1.0 / 3.6:
            speed_profile[-i] = 1.0 / 3.6

    return speed_profile


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def generate_path(s):
    # max_c = math.tan(C.MAX_STEER) / C.WB
    max_c = math.tan(0.25) / C.WB
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


def main():
    # generate path
    ax = np.arange(0, 50, 0.5)
    ay = [math.sin(ix / 5.0) * ix / 2.0 for ix in ax]

    cx, cy, cyaw, ck, _ = cs.calc_spline_course(ax, ay, ds=C.dt)

    t = 0.0
    maxTime = 100.0
    yaw_old = 0.0
    x0, y0, yaw0 = cx[0], cy[0], cyaw[0]
    xrec, yrec, yawrec = [], [], []

    node = Node(x=x0, y=y0, yaw=yaw0, v=0.0)
    ref_trajectory = Trajectory(cx, cy, cyaw, ck)
    e, e_th = 0.0, 0.0

    while t < maxTime:
        speed_ref = 25.0 / 3.6
        sp = calc_speed_profile(cyaw, speed_ref)

        dl, target_ind, e, e_th, ai = ref_trajectory.lqr_control(
            node, e, e_th, sp, C.lqr_Q, C.lqr_R)

        dist = math.hypot(node.x - cx[-1], node.y - cy[-1])
        node.update(ai, dl, 1.0)
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
        plt.plot(cx[target_ind], cy[target_ind], '.r')
        draw.draw_car(node.x, node.y, node.yaw, steer, C)
        plt.axis("equal")
        plt.title("FrontWheelFeedback: v=" + str(node.v * 3.6)[:4] + "km/h")
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    main()
