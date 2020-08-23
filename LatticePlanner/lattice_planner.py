import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

from CurvesGenerator import cubic_spline, quintic_polynomial, quartic_polynomial
import LatticePlanner.env as env
import LatticePlanner.draw as draw


class C:
    # Parameter
    MAX_SPEED = 50.0 / 3.6
    MAX_ACCEL = 8.0
    MAX_CURVATURE = 2.5

    ROAD_WIDTH = 8.0
    ROAD_SAMPLE_STEP = 1.0

    T_STEP = 0.15
    MAX_T = 5.0
    MIN_T = 4.0

    TARGET_SPEED = 30.0 / 3.6
    SPEED_SAMPLE_STEP = 5.0 / 3.6

    # cost weights for Cruising
    K_JERK = 0.1
    K_TIME = 0.1
    K_V_DIFF = 1.0
    K_OFFSET = 1.5
    K_COLLISION = 500

    # cost weights for Stopping
    # K_JERK = 0.1
    # K_TIME = 0.1
    # K_V_DIFF = 200
    # K_OFFSET = 1.0
    # K_COLLISION = 500

    # parameters for vehicle
    K_SIZE = 0.9
    RF = 4.5 * K_SIZE  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0 * K_SIZE  # [m] distance from rear to vehicle back end of vehicle
    W = 3.0 * K_SIZE  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 3.5 * K_SIZE  # [m] Wheel base
    TR = 0.5 * K_SIZE  # [m] Tyre radius
    TW = 1 * K_SIZE  # [m] Tyre width
    MAX_STEER = 0.6  # [rad] maximum steering angle


class Path:
    def __init__(self):
        self.t = []

        self.l = []
        self.l_v = []
        self.l_a = []
        self.l_jerk = []

        self.s = []
        self.s_v = []
        self.s_a = []
        self.s_jerk = []

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.curv = []

        self.cost = 0.0


def sampling_paths_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    PATHS = dict()

    for s1_v in np.arange(C.TARGET_SPEED * 0.6, C.TARGET_SPEED * 1.4, C.TARGET_SPEED * 0.2):

        for t1 in np.arange(4.5, 5.5, 0.2):
            path_pre = Path()
            path_lon = quartic_polynomial.QuarticPolynomial(s0, s0_v, s0_a, s1_v, 0.0, t1)

            path_pre.t = list(np.arange(0.0, t1, C.T_STEP))
            path_pre.s = [path_lon.calc_xt(t) for t in path_pre.t]
            path_pre.s_v = [path_lon.calc_dxt(t) for t in path_pre.t]
            path_pre.s_a = [path_lon.calc_ddxt(t) for t in path_pre.t]
            path_pre.s_jerk = [path_lon.calc_dddxt(t) for t in path_pre.t]

            for l1 in np.arange(-C.ROAD_WIDTH, C.ROAD_WIDTH, C.ROAD_SAMPLE_STEP):
                path = copy.deepcopy(path_pre)
                path_lat = quintic_polynomial.QuinticPolynomial(l0, l0_v, l0_a, l1, 0.0, 0.0, t1)

                path.l = [path_lat.calc_xt(t) for t in path_pre.t]
                path.l_v = [path_lat.calc_dxt(t) for t in path_pre.t]
                path.l_a = [path_lat.calc_ddxt(t) for t in path_pre.t]
                path.l_jerk = [path_lat.calc_dddxt(t) for t in path_pre.t]

                path.x, path.y = SL_2_XY(path.s, path.l, ref_path)
                path.yaw, path.curv, path.ds = calc_yaw_curv(path.x, path.y)

                l_jerk_sum = sum(np.abs(path.l_jerk))
                s_jerk_sum = sum(np.abs(path.s_jerk))
                v_diff = abs(C.TARGET_SPEED - path.s_v[-1])

                path.cost = C.K_JERK * (l_jerk_sum + s_jerk_sum) + \
                            C.K_V_DIFF * v_diff + \
                            C.K_TIME * t1 * 2 + \
                            C.K_OFFSET * abs(path.l[-1]) + \
                            C.K_COLLISION * is_path_collision(path)

                PATHS[path] = path.cost

    return PATHS


def sampling_paths_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    PATHS = dict()

    for s1_v in [-1.0, 0.0, 1.0, 2.0]:

        for t1 in np.arange(1.0, 16.0, 1.0):
            path_pre = Path()
            path_lon = quintic_polynomial.QuinticPolynomial(s0, s0_v, s0_a, 55.0, s1_v, 0.0, t1)

            path_pre.t = list(np.arange(0.0, t1, C.T_STEP))
            path_pre.s = [path_lon.calc_xt(t) for t in path_pre.t]
            path_pre.s_v = [path_lon.calc_dxt(t) for t in path_pre.t]
            path_pre.s_a = [path_lon.calc_ddxt(t) for t in path_pre.t]
            path_pre.s_jerk = [path_lon.calc_dddxt(t) for t in path_pre.t]

            for l1 in [0.0]:
                path = copy.deepcopy(path_pre)
                path_lat = quintic_polynomial.QuinticPolynomial(l0, l0_v, l0_a, l1, 0.0, 0.0, t1)

                path.l = [path_lat.calc_xt(t) for t in path_pre.t]
                path.l_v = [path_lat.calc_dxt(t) for t in path_pre.t]
                path.l_a = [path_lat.calc_ddxt(t) for t in path_pre.t]
                path.l_jerk = [path_lat.calc_dddxt(t) for t in path_pre.t]

                path.x, path.y = SL_2_XY(path.s, path.l, ref_path)
                path.yaw, path.curv, path.ds = calc_yaw_curv(path.x, path.y)

                if path.yaw is None:
                    continue

                l_jerk_sum = sum(np.abs(path.l_jerk))
                s_jerk_sum = sum(np.abs(path.s_jerk))
                v_diff = (path.s_v[-1]) ** 2

                path.cost = C.K_JERK * (l_jerk_sum + s_jerk_sum) + \
                            C.K_V_DIFF * v_diff + \
                            C.K_TIME * t1 * 2 + \
                            C.K_OFFSET * abs(path.l[-1]) + \
                            50.0 * sum(np.abs(path.s_v))

                PATHS[path] = path.cost

    return PATHS


def SL_2_XY(s_set, l_set, ref_path):
    pathx, pathy = [], []

    for i in range(len(s_set)):
        x_ref, y_ref = ref_path.calc_position(s_set[i])

        if x_ref is None:
            break

        yaw = ref_path.calc_yaw(s_set[i])
        x = x_ref + l_set[i] * math.cos(yaw + math.pi / 2.0)
        y = y_ref + l_set[i] * math.sin(yaw + math.pi / 2.0)

        pathx.append(x)
        pathy.append(y)

    return pathx, pathy


def calc_yaw_curv(x, y):
    yaw, curv, ds = [], [], []

    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        ds.append(math.hypot(dx, dy))
        yaw.append(math.atan2(dy, dx))

    if len(yaw) == 0:
        return None, None, None

    yaw.append(yaw[-1])
    ds.append(ds[-1])

    for i in range(len(yaw) - 1):
        curv.append((yaw[i + 1] - yaw[i]) / ds[i])

    return yaw, curv, ds


def is_path_collision(path):
    index = range(0, len(path.x), 5)
    x = [path.x[i] for i in index]
    y = [path.y[i] for i in index]
    yaw = [path.yaw[i] for i in index]

    for ix, iy, iyaw in zip(x, y, yaw):
        d = 1.8
        dl = (C.RF - C.RB) / 2.0
        r = math.hypot((C.RF + C.RB) / 2.0, C.W / 2.0) + d

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        for i in range(len(C.obs)):
            xo = C.obs[i][0] - cx
            yo = C.obs[i][1] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return 1.0

    return 0.0


def verify_path(path):
    # if any([v > C.speed_max for v in path.s_v]) or \
    #         any([abs(a) > C.acceleration_max for a in path.s_a]):
    #     return False

    if any([v > C.MAX_SPEED for v in path.s_v]) or \
            any([abs(a) > C.MAX_ACCEL for a in path.s_a]) or \
            any([abs(curv) > C.MAX_CURVATURE for curv in path.curv]):
        return False

    return True


def extract_optimal_path(paths):
    if len(paths) == 0:
        return None

    while len(paths) > 1:
        path = min(paths, key=paths.get)
        paths.pop(path)
        if verify_path(path) is False:
            continue
        else:
            return path

    return paths[-1]


def lattice_planner_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    paths = sampling_paths_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)
    path = extract_optimal_path(paths)

    return path


def lattice_planner_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path):
    paths = sampling_paths_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)
    path = extract_optimal_path(paths)

    return path


def get_reference_line(x, y):
    index = range(0, len(x), 3)
    x = [x[i] for i in index]
    y = [y[i] for i in index]

    cubicspline = cubic_spline.Spline2D(x, y)
    s = np.arange(0, cubicspline.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = cubicspline.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(cubicspline.calc_yaw(i_s))
        rk.append(cubicspline.calc_curvature(i_s))

    return rx, ry, ryaw, rk, cubicspline


def pi_2_pi(theta):
    if theta > math.pi:
        return theta - 2.0 * math.pi

    if theta < -math.pi:
        return theta + 2.0 * math.pi

    return theta


def main_Crusing():
    ENV = env.ENVCrusing()
    wx, wy = ENV.ref_line
    bx1, by1 = ENV.bound_in
    bx2, by2 = ENV.bound_out

    C.obs = np.array([[50, 10], [96, 25], [70, 40],
                      [40, 50], [25, 75]])

    obs_x = [x for x, y in C.obs]
    obs_y = [y for x, y in C.obs]

    rx, ry, ryaw, rk, ref_path = get_reference_line(wx, wy)

    l0 = 2.0  # current lateral position [m]
    l0_v = 0.0  # current lateral speed [m/s]
    l0_a = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    s0_v = 20.0 / 3.6  # current speed [m/s]
    s0_a = 0.0

    while True:
        path = lattice_planner_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)

        if path is None:
            print("No feasible path found!!")
            break

        l0 = path.l[1]
        l0_v = path.l_v[1]
        l0_a = path.l_a[1]
        s0 = path.s[1]
        s0_v = path.s_v[1]
        s0_a = path.s_a[1]

        if np.hypot(path.x[1] - rx[-1], path.y[1] - ry[-1]) <= 2.0:
            print("Goal")
            break

        dy = (path.yaw[2] - path.yaw[1]) / path.ds[1]
        steer = pi_2_pi(math.atan(-C.WB * dy))

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(rx, ry, linestyle='--', color='gray')
        plt.plot(bx1, by1, linewidth=1.5, color='k')
        plt.plot(bx2, by2, linewidth=1.5, color='k')
        plt.plot(path.x[1:], path.y[1:], linewidth='2', color='royalblue')
        plt.plot(obs_x, obs_y, 'ok')
        draw.draw_car(path.x[1], path.y[1], path.yaw[1], steer, C)
        plt.title("[Crusing Mode]  v :" + str(s0_v * 3.6)[0:4] + " km/h")
        plt.axis("equal")
        plt.pause(0.0001)

    plt.pause(0.0001)
    plt.show()


def main_Stopping():
    ENV = env.ENVStopping()
    wx, wy = ENV.ref_line
    bx1, by1 = ENV.bound_up
    bx2, by2 = ENV.bound_down

    C.ROAD_WIDTH = ENV.road_width
    rx, ry, ryaw, rk, ref_path = get_reference_line(wx, wy)

    l0 = 0.0  # current lateral position [m]
    l0_v = 0.0  # current lateral speed [m/s]
    l0_a = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    s0_v = 30.0 / 3.6  # current speed [m/s]
    s0_a = 0.0

    while True:
        path = lattice_planner_for_Stopping(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)

        if path is None:
            print("No feasible path found!!")
            break

        l0 = path.l[1]
        l0_v = path.l_v[1]
        l0_a = path.l_a[1]
        s0 = path.s[1]
        s0_v = path.s_v[1]
        s0_a = path.s_a[1]

        if np.hypot(path.x[1] - 56.0, path.y[1] - 0) <= 1.5:
            print("Goal")
            break

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(rx, ry, linestyle='--', color='gray')
        plt.plot(bx1, by1, linewidth=1.5, color='k')
        plt.plot(bx2, by2, linewidth=1.5, color='k')
        plt.plot(path.x[1:], path.y[1:], linewidth='2', color='royalblue')
        draw.draw_car(path.x[1], path.y[1], path.yaw[1], 0.0, C)
        plt.title("[Stopping Mode]  v :" + str(s0_v * 3.6)[0:4] + " km/h")
        plt.axis("equal")
        plt.pause(0.0001)

    plt.pause(0.0001)
    plt.show()

    plt.plot(rx, ry, linestyle='--', color='gray')
    plt.plot(bx1, by1, linewidth=1.5, color='k')
    plt.plot(bx2, by2, linewidth=1.5, color='k')
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main_Crusing()
    # main_Stopping()
