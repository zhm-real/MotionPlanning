"""
LQR controller for autonomous vehicle
@author: huiming zhou (zhou.hm0420@gmail.com)

This controller is the python version of LQR controller of Apollo.
GitHub link of BaiDu Apollo: https://github.com/ApolloAuto/apollo

In this file, we will use hybrid A* planner for path planning, while using
LQR with parameters in Apollo as controller for path tracking.
"""

import os
import sys
import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import HybridAstarPlanner.hybrid_astar as HybridAStar
import HybridAstarPlanner.draw as draw
from Control.lateral_controller_conf import *


class Gear(Enum):
    GEAR_DRIVE = 1
    GEAR_REVERSE = 2


class VehicleState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0,
                 v=0.0, gear=Gear.GEAR_DRIVE):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.gear = gear


class TrajectoryAnalyzer:
    def __init__(self, x, y, yaw, k):
        self.x_ = x
        self.y_ = y
        self.yaw_ = yaw
        self.k_ = k

        self.ind_old = 0
        self.ind_end = len(x)

    def ToTrajectoryFrame(self, vehicle_state):
        """
        errors to trajectory frame

        theta_e = yaw_vehicle - yaw_ref_path
        e_cg = lateral distance of center of gravity (cg) in frenet frame

        :param vehicle_state: vehicle state (class VehicleState)
        :return: theta_e, e_cg, yaw_ref, k_ref
        """

        x_cg = vehicle_state.x
        y_cg = vehicle_state.y
        yaw = vehicle_state.yaw

        # calc nearest point in ref path
        dx = [x_cg - ix for ix in self.x_[self.ind_old: self.ind_end]]
        dy = [y_cg - iy for iy in self.y_[self.ind_old: self.ind_end]]

        ind_add = int(np.argmin(np.hypot(dx, dy)))
        dist = math.hypot(dx[ind_add], dy[ind_add])

        # calc relative position of vehicle to ref path
        vec_axle_rot_90 = np.array([[math.cos(yaw + math.pi / 2.0)],
                                    [math.sin(yaw + math.pi / 2.0)]])

        vec_path_2_cg = np.array([[dx[ind_add] / dist],
                                  [dy[ind_add] / dist]])

        if np.dot(vec_axle_rot_90.T, vec_path_2_cg) > 0.0:
            e_cg = 1.0 * dist  # vehicle on the right of ref path
        else:
            e_cg = -1.0 * dist  # vehicle on the left of ref path

        # calc yaw error: theta_e = yaw_vehicle - yaw_ref
        self.ind_old += ind_add
        yaw_ref = self.yaw_[self.ind_old]
        theta_e = pi_2_pi(yaw - yaw_ref)

        # calc ref curvature
        k_ref = self.k_[self.ind_old]

        return theta_e, e_cg, yaw_ref, k_ref


class LatController:
    def __init__(self):
        self.vehicle_state = VehicleState()

    def UpdateMatrix(self):
        """
        calc A and b matrices of linearized, discrete system.
        :return: A, b
        """

        ts_ = ts
        mass_ = m_f + m_r

        v = self.vehicle_state.v

        matrix_a_ = np.zeros((state_size, state_size))  # continuous A matrix

        if self.vehicle_state.gear == Gear.GEAR_REVERSE:
            """
            A matrix (Gear Reverse)
            [0.0, 0.0, 1.0 * v 0.0;
             0.0, -(c_f + c_r) / m / v, (c_f + c_r) / m,
             (l_r * c_r - l_f * c_f) / m / v;
             0.0, 0.0, 0.0, 1.0;
             0.0, (lr * cr - lf * cf) / i_z / v, (l_f * c_f - l_r * c_r) / i_z,
             -1.0 * (l_f^2 * c_f + l_r^2 * c_r) / i_z / v;]
            """

            matrix_a_[0][1] = 0.0
            matrix_a_[0][2] = 1.0 * v
        else:
            """
            A matrix (Gear Drive)
            [0.0, 1.0, 0.0, 0.0;
             0.0, -(c_f + c_r) / m / v, (c_f + c_r) / m,
             (l_r * c_r - l_f * c_f) / m / v;
             0.0, 0.0, 0.0, 1.0;
             0.0, (lr * cr - lf * cf) / i_z / v, (l_f * c_f - l_r * c_r) / i_z,
             -1.0 * (l_f^2 * c_f + l_r^2 * c_r) / i_z / v;]
            """

            matrix_a_[0][1] = 1.0
            matrix_a_[0][2] = 0.0

        matrix_a_[1][1] = -1.0 * (c_f + c_r) / mass_ / v
        matrix_a_[1][2] = (c_f + c_r) / mass_
        matrix_a_[1][3] = (l_f * c_f - l_f * c_f) / mass_ / v
        matrix_a_[2][3] = 1.0
        matrix_a_[3][1] = (l_r * c_r - l_f * c_f) / Iz / v
        matrix_a_[3][2] = (l_f * c_f - l_r * c_r) / Iz
        matrix_a_[3][3] = -1.0 * (l_f ** 2 * c_f + l_r ** 2 * c_r) / Iz / v

        # Tustin's method (bilinear transform)
        matrix_i = np.eye(state_size)  # identical matrix
        matrix_ad_ = np.linalg.pinv(matrix_i - ts_ * 0.5 * matrix_a_) * \
                     (matrix_i + ts_ * 0.5 * matrix_a_)  # discrete A matrix

        # b = [0.0, c_f / m, 0.0, l_f * c_f / I_z].T
        matrix_b_ = np.zeros((state_size, 1))  # continuous b matrix
        matrix_b_[1][0] = c_f / mass_
        matrix_b_[3][0] = l_f * c_f / Iz
        matrix_bd_ = matrix_b_ * ts_  # discrete b matrix

        return matrix_ad_, matrix_bd_


def pi_2_pi(angle):
    """
    regulate theta to -pi ~ pi.
    :param angle: input angle
    :return: regulated angle
    """

    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

def main():
    return


if __name__ == '__main__':
    main()
