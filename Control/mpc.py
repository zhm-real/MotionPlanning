import os
import sys
import cvxpy
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import CurvesGenerator.cubic_spline as cs
import Control.draw as draw


NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 10  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

MAX_ITER = 5  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 20.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
K = 1.5
RF = 3.3 * K
RB = 0.8 * K
L = RF + RB
W = 2.5 * K
WD = 0.7 * W
WB = 2.5 * K
TR = 0.44 * K
TW = 0.7 * K

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 2.0  # maximum accel [m/ss]


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None

    def update_state(self, a, delta):
        delta = min(max(delta, MIN_SPEED), MAX_SPEED)

        self.x = self.x + self.v * math.cos(self.yaw) * DT
        self.y = self.y + self.v * math.sin(self.yaw) * DT
        self.yaw = self.yaw + self.v / WB * math.tan(delta) * DT
        self.v = min(max(self.v + a * DT, MIN_SPEED), MAX_TIME)


def pi_2_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi

    while angle < -math.pi:
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(v, phi, delta):

    A = np.array([[1.0, 0.0, DT * math.cos(phi), - DT * v * math.sin(phi)],
                  [0.0, 1.0, DT * math.sin(phi), DT * v * math.cos(phi)],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, DT * math.tan(delta) / WB, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [DT, 0.0],
                  [0.0, DT * v / (WB * math.cos(delta) ** 2)]])

    C = np.array([DT * v * math.sin(phi) * phi,
                  - DT * v * math.cos(phi) * phi,
                  0.0,
                  - DT * v * delta / (WB * math.cos(delta) ** 2)])

    return A, B, C


def plot_car(x, y, yaw, steer, color='black'):
    car = np.array([[-RB, -RB, RF, RF, -RB],
                    [W / 2, -W / 2, -W / 2, W / 2, W / 2]])

    wheel = np.array([[-TR, -TR, TR, TR, -TR],
                      [TW / 4, -TW / 4, -TW / 4, TW / 4, TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[WB], [-WD / 2]])
    flWheel += np.array([[WB], [WD / 2]])
    rrWheel[1, :] -= WD / 2
    rlWheel[1, :] += WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = np.hypot(dx, dy)
    mind = min(d)
    ind = np.argmin(d) + pind

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, a, delta, xref):
    xpred = xref * 0.0
    for i in range(len(x0)):
        xpred[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])

    for (ai, di, i) in zip(a, delta, range(1, T + 1)):
        state.update_state(ai, di)
        xpred[0, i] = state.x
        xpred[1, i] = state.y
        xpred[2, i] = state.v
        xpred[3, i] = state.yaw

    return xpred


def mpc_control(xref, x0, dref, oa, odelta):

    for i in range(MAX_ITER):
        xpred = predict_motion(x0, oa, odelta, xref)
        poa, pod = oa[:], odelta[:]
        oa, odelta, ox, oy, oyaw, ov = solve_cvx(xref, xpred, x0, dref)

        if sum(abs(oa - poa)) <= DU_TH and \
                sum(abs(odelta - pod)) <= DU_TH:
            break

    return oa, odelta, ox, oy, oyaw, ov


def solve_cvx(xref, xpred, x0, dref):
    """
    solve cvx

    xref: reference point
    xpred: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xpred[2, t], xpred[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    dist = 0.0

    for i in range(T + 1):
        dist += abs(state.v) * DT
        dind = int(round(dist / dl))
        index = min(ind + dind, len(cx) - 1)

        xref[0, i] = cx[index]
        xref[1, i] = cy[index]
        xref[2, i] = sp[index]
        xref[3, i] = cyaw[index]
        dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):

    d = math.hypot(state.x - goal[0], state.y - goal[1])

    if d <= GOAL_DIS and abs(tind - nind) < 5 and \
            abs(state.v) <= STOP_SPEED:
        return True

    return False


def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        speed_profile[i] = direction * target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def main():
    print(__file__ + " start!!")

    dl = 1.0  # course tick
    ax = [0.0, 15.0, 30.0, 45.0, 60.0, 70.0]
    ay = [0.0, 45.0, 15.0, 30.0, 0.0, 10.0]
    cx, cy, cyaw, ck, _ = cs.calc_spline_course(ax, ay, ds=dl)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    goal = [cx[-1], cy[-1]]

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    oa = [0.0] * T
    odelta = [0.0] * T

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = mpc_control(
            xref, x0, dref, oa, odelta)

        if odelta is not None:
            di, ai = odelta[0], oa[0]

        state.update_state(ai, di)
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event:
                                     [exit(0) if event.key == 'escape' else None])
        if ox is not None:
            plt.plot(ox, oy, "xr", label="MPC")
        plt.plot(cx, cy, "-r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
        plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        plot_car(state.x, state.y, state.yaw, steer=di)
        plt.axis("equal")
        plt.grid(True)
        plt.title("Time[s]:" + str(round(time, 2))
                  + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
        plt.pause(0.0001)

    plt.show()


if __name__ == '__main__':
    main()
