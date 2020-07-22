import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

from CurvesGenerator import cubic_spline, quintic_polynomial, quartic_polynomial
import LatticePlanner.env as env


class C:
    # Parameter
    MAX_SPEED = 50.0 / 3.6
    MAX_ACCEL = 10.0
    MAX_CURVATURE = 5.0

    ROAD_WIDTH = 8.0
    ROAD_SAMPLE_STEP = 2.0

    T_STEP = 0.2
    MAX_T = 5.0
    MIN_T = 4.0

    TARGET_SPEED = 30.0 / 3.6
    SPEED_SAMPLE_STEP = 5.0 / 3.6

    # cost weights
    K_JERK = 0.1
    K_TIME = 0.1
    K_V_DIFF = 1.0
    K_OFFSET = 1.0

    # parameters for vehicle
    RF = 4.5  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0  # [m] distance from rear to vehicle back end of vehicle
    W = 3.0  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 3.5  # [m] Wheel base
    TR = 0.5  # [m] Tyre radius
    TW = 1  # [m] Tyre width
    MAX_STEER = 0.6  # [rad] maximum steering angle

    obs = [[60.0, 40.0], [10.0, 60.0]]
    obs_tree = kd.KDTree(obs)


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

    for s1_v in np.arange(C.TARGET_SPEED - 2 * C.SPEED_SAMPLE_STEP,
                          C.TARGET_SPEED + 2 * C.SPEED_SAMPLE_STEP, C.SPEED_SAMPLE_STEP):

        for t1 in np.arange(4.0, 5.0, 0.2):
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

                l_jerk_sum = sum(np.power(path.l_jerk, 2))
                s_jerk_sum = sum(np.power(path.s_jerk, 2))

                v_diff = (C.TARGET_SPEED - path.s_v[-1]) ** 2

                path.cost = C.K_JERK * (l_jerk_sum + s_jerk_sum) + \
                            C.K_V_DIFF * (path.l_v[-1] ** 2 + v_diff) + \
                            C.K_TIME * t1 * 2 + \
                            C.K_OFFSET * abs(path.l[-1])

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
        d = 0.3
        dl = (C.RF - C.RB) / 2.0
        r = (C.RF + C.RB) / 2.0 + d

        cx = ix + dl * math.cos(iyaw)
        cy = iy + dl * math.sin(iyaw)

        ids = C.obs_tree.query_ball_point([cx, cy], r)

        if not ids:
            continue

        for i in ids:
            xo = C.obs[i][0] - cx
            yo = C.obs[i][1] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return True

    return False


def verify_path(path):
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


def get_reference_line(x, y):
    index = range(0, len(x), 2)
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


def main():
    ENV = env.ENV()
    wx, wy = ENV.design_reference_line()
    bx1, by1 = ENV.design_boundary_in()
    bx2, by2 = ENV.design_boundary_out()

    rx, ry, ryaw, rk, ref_path = get_reference_line(wx, wy)

    l0 = 2.0  # current lateral position [m]
    l0_v = 0.0  # current lateral speed [m/s]
    l0_a = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position
    s0_v = 20.0 / 3.6  # current speed [m/s]
    s0_a = 0.0

    for i in range(1000):
        path = lattice_planner_for_Cruising(l0, l0_v, l0_a, s0, s0_v, s0_a, ref_path)

        if path is None:
            print("No feasible path found!!")
            break

        l0 = path.l[1]
        l0_v = path.l_v[1]
        l0_a = path.l_a[1]
        s0 = path.s[1]
        s0_v = path.s_v[1]
        s0_a = 0.0

        if np.hypot(path.x[1] - rx[-1], path.y[1] - ry[-1]) <= 3.0:
            print("Goal")
            break

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        plt.plot(rx, ry)
        plt.plot(bx1, by1, linewidth=1.5, color='k')
        plt.plot(bx2, by2, linewidth=1.5, color='k')
        # plt.plot(C.obs[:, 0], C.obs[:, 1], "xk")
        plt.plot(path.x[1:], path.y[1:], marker='.', color='red')
        plt.plot(path.x[1], path.y[1], "vc")
        plt.title("v[km/h]:" + str(s0_v * 3.6)[0:4])
        plt.axis("equal")
        plt.pause(0.0001)

    plt.pause(0.0001)
    plt.show()


if __name__ == '__main__':
    main()
