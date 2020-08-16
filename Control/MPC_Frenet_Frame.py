"""
Linear MPC controller (Frenet frame)
author: huiming zhou
"""

import os
import sys
import math
import cvxpy
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs


class P:
    # System config
    NX = 5  # state vector: z = [e, e_dot, theta_e, theta_e_dot, v]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 6  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 10.0 / 3.6  # target speed
    N_IND = 10  # search index number
    dt = 0.2  # time step
    d_dist = 1.0  # dist step
    du_res = 0.1  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = -20.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 1.0  # maximum acceleration [m/s2]


class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct

    def update(self, a, delta, direct):
        delta = self.limit_input_delta(delta)
        self.x += self.v * math.cos(self.yaw) * P.dt
        self.y += self.v * math.sin(self.yaw) * P.dt
        self.yaw += self.v / P.WB * math.tan(delta) * P.dt
        self.direct = direct
        self.v += self.direct * a * P.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= P.steer_max:
            return P.steer_max

        if delta <= -P.steer_max:
            return -P.steer_max

        return delta

    @staticmethod
    def limit_speed(v):
        if v >= P.speed_max:
            return P.speed_max

        if v <= P.speed_min:
            return P.speed_min

        return v


class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)
        self.ind_old = 0

    def calc_theta_e_and_er(self, node):
        dx = [node.x - x for x in self.cx[self.ind_old: (self.ind_old + P.N_IND)]]
        dy = [node.y - y for y in self.cy[self.ind_old: (self.ind_old + P.N_IND)]]
        dist = np.hypot(dx, dy)

        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        theta = node.yaw
        theta_p = self.cyaw[ind]
        theta_e = pi_2_pi(theta - theta_p)

        return theta_e, er, ind


def calc_ref_trajectory_in_T_step(node, ref_path, sp):
    z_ref = np.zeros((P.NX, P.T + 1))
    length = ref_path.length

    theta_e, er, ind = ref_path.calc_theta_e_and_er(node)

    z_ref[4, 0] = sp[ind]
    dist_move = 0.0

    for i in range(1, P.T + 1):
        dist_move += abs(node.v) * P.dt
        ind_move = int(round(dist_move / P.d_dist))
        index = min(ind + ind_move, length - 1)

        z_ref[4, i] = sp[index]

    return z_ref, ind, theta_e, er


def linear_mpc_control(z_ref, node0, z0, a_old, delta_old):
    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T

    for k in range(P.iter_max):
        v_bar = predict_states_in_T_step(node0, a_old, delta_old)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old = solve_linear_mpc(z_ref, v_bar, z0)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
            break

    return a_old, delta_old


def solve_linear_mpc(z_ref, v_bar, z0):
    z = cvxpy.Variable((P.NX, P.T + 1))
    u = cvxpy.Variable((P.NU, P.T))

    cost = 0.0
    constrains = []

    for t in range(P.T):
        cost += cvxpy.quad_form(u[:, t], P.R)
        cost += cvxpy.quad_form(z[:, t] - z_ref[:, t], P.Q)

        A, B = calc_linear_discrete_model(v_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t]]

        if t < P.T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

    cost += cvxpy.quad_form(z_ref[:, P.T] - z[:, P.T], P.Qf)

    constrains += [z[:, 0] == z0]
    constrains += [z[4, :] <= P.speed_max]
    constrains += [z[4, :] >= P.speed_min]
    constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    prob.solve(solver=cvxpy.OSQP)

    a, delta = None, None

    if prob.status == cvxpy.OPTIMAL or \
            prob.status == cvxpy.OPTIMAL_INACCURATE:
        a = u.value[0, :]
        delta = u.value[1, :]
    else:
        print("Cannot solve linear mpc!")

    print(delta)

    return a, delta


def predict_states_in_T_step(node0, a, delta):
    v_bar = [0.0] * (P.T + 1)
    v_bar[0] = node0.v

    for ai, di, i in zip(a, delta, range(1, P.T + 1)):
        node0.update(ai, di, 1.0)
        v_bar[i] = node0.v

    return v_bar


def calc_linear_discrete_model(v):
    A = np.array([[1.0, P.dt, 0.0, 0.0, 0.0],
                  [0.0, 0.0, v, 0.0, 0.0],
                  [0.0, 0.0, 1.0, P.dt, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [0.0, 0.0],
                  [v / P.WB, 0.0],
                  [0.0, P.dt]])

    return A, B


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

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


def main():
    ax = [0.0, 20.0, 40.0, 55.0, 70.0, 85.0]
    ay = [0.0, 50.0, 20.0, 35.0, 0.0, 10.0]

    cx, cy, cyaw, ck, s = \
        cs.calc_spline_course(ax, ay, ds=P.d_dist)

    sp = calc_speed_profile(cx, cy, cyaw, P.target_speed)

    ref_path = PATH(cx, cy, cyaw, ck)
    node = Node(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    time = 0.0
    x = [node.x]
    y = [node.y]
    yaw = [node.yaw]
    v = [node.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]

    delta_opt, a_opt = None, None
    a_exc, delta_exc = 0.0, 0.0

    while time < P.time_max:
        z_ref, target_ind, theta_e, er = \
            calc_ref_trajectory_in_T_step(node, ref_path, sp)

        node0 = Node(x=node.x, y=node.y, yaw=node.yaw, v=node.v)
        z0 = [er, 0.0, theta_e, 0.0, node.v]

        a_opt, delta_opt = \
            linear_mpc_control(z_ref, node0, z0, a_opt, delta_opt)

        # node_opt = Node(x=node.x, y=node.y, yaw=node.yaw, v=node.v)
        # x_opt, y_opt = [node_opt.x], [node_opt.y]

        if delta_opt is not None:
            delta_exc, a_exc = delta_opt[0], a_opt[0]

            # for ao, do in zip(a_opt, delta_opt):
            #     node_opt.update(ao, do, 1.0)
            #     x_opt.append(node_opt.x)
            #     y_opt.append(node_opt.y)

        node.update(a_exc, delta_exc, 1.0)
        time += P.dt

        x.append(node.x)
        y.append(node.y)
        yaw.append(node.yaw)
        v.append(node.v)
        t.append(time)
        d.append(delta_exc)
        a.append(a_exc)

        dist = math.hypot(node.x - cx[-1], node.y - cy[-1])

        if dist < P.dist_stop and \
                abs(node.v) < P.speed_stop:
            break

        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event:
                                     [exit(0) if event.key == 'escape' else None])

        # if x_opt is not None:
        #     plt.plot(x_opt, y_opt, 'xr')

        plt.plot(cx, cy, '-r')
        plt.plot(x, y, '-b')
        plt.plot(z_ref[0, :], z_ref[1, :], 'xk')
        plt.plot(cx[target_ind], cy[target_ind], 'xg')
        plt.axis("equal")
        plt.title("Linear MPC, " + "v = " + str(round(node.v * 3.6, 2)))
        plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    main()