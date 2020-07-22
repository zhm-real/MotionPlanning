"""
Environment for Lattice Planner Simulation
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import CurvesGenerator.reeds_shepp as rs


class ENV:
    def __init__(self):
        self.max_c = 0.15
        self.road_width = 8.0
        # self.ref_line = self.design_reference_line()
        # self.obs = self.design_obstacles()

    def design_reference_line(self):
        rx, ry, ryaw, rc = [], [], [], []
        step_curve = 0.15 * math.pi
        step_line = 8

        cx, cy, cr = 30, 30, 20
        theta = np.arange(math.pi, math.pi * 1.5, step_curve)
        for itheta in theta:
            rx.append(cx + cr * math.cos(itheta))
            ry.append(cy + cr * math.sin(itheta))

        for ix in np.arange(30, 80, step_line):
            rx.append(ix)
            ry.append(10)

        cx, cy, cr = 80, 25, 15
        theta = np.arange(-math.pi / 2.0, math.pi / 2.0, step_curve)
        for itheta in theta:
            rx.append(cx + cr * math.cos(itheta))
            ry.append(cy + cr * math.sin(itheta))

        for ix in np.arange(80, 60, -step_line):
            rx.append(ix)
            ry.append(40)

        cx, cy, cr = 60, 60, 20
        theta = np.arange(-math.pi / 2.0, -math.pi, -step_curve)
        for itheta in theta:
            rx.append(cx + cr * math.cos(itheta))
            ry.append(cy + cr * math.sin(itheta))

        cx, cy, cr = 25, 60, 15
        theta = np.arange(0.0, math.pi, step_curve)
        for itheta in theta:
            rx.append(cx + cr * math.cos(itheta))
            ry.append(cy + cr * math.sin(itheta))

        for iy in np.arange(60, 30, -step_line):
            rx.append(10)
            ry.append(iy)

        # rx.append(rx[0])
        # ry.append(ry[0])

        return rx, ry

    def design_boundary_in(self):
        bx, by = [], []
        step_curve = 0.1
        step_line = 2
        road_width = self.road_width

        cx, cy, cr = 30, 30, 20 - road_width
        theta = np.arange(math.pi, math.pi * 1.5, step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        for ix in np.arange(30, 80, step_line):
            bx.append(ix)
            by.append(10 + road_width)

        cx, cy, cr = 80, 25, 15 - road_width
        theta = np.arange(-math.pi / 2.0, math.pi / 2.0, step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        for ix in np.arange(80, 60, -step_line):
            bx.append(ix)
            by.append(40 - road_width)

        cx, cy, cr = 60, 60, 20 + road_width
        theta = np.arange(-math.pi / 2.0, -math.pi, -step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        cx, cy, cr = 25, 60, 15 - road_width
        theta = np.arange(0.0, math.pi, step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        for iy in np.arange(60, 30, -step_line):
            bx.append(10 + road_width)
            by.append(iy)

        bx.append(bx[0])
        by.append(by[0])

        return bx, by

    def design_boundary_out(self):
        bx, by = [], []
        step_curve = 0.1
        step_line = 2
        road_width = self.road_width

        cx, cy, cr = 30, 30, 20 + road_width
        theta = np.arange(math.pi, math.pi * 1.5, step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        for ix in np.arange(30, 80, step_line):
            bx.append(ix)
            by.append(10 - road_width)

        cx, cy, cr = 80, 25, 15 + road_width
        theta = np.arange(-math.pi / 2.0, math.pi / 2.0, step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        for ix in np.arange(80, 60, -step_line):
            bx.append(ix)
            by.append(40 + road_width)

        cx, cy, cr = 60, 60, 20 - road_width
        theta = np.arange(-math.pi / 2.0, -math.pi, -step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        cx, cy, cr = 25, 60, 15 + road_width
        theta = np.arange(0.0, math.pi, step_curve)
        for itheta in theta:
            bx.append(cx + cr * math.cos(itheta))
            by.append(cy + cr * math.sin(itheta))

        for iy in np.arange(60, 30, -step_line):
            bx.append(10 - road_width)
            by.append(iy)

        bx.append(bx[0])
        by.append(by[0])

        return bx, by

    def design_obstacles(self):
        max_c = self.max_c
        obs = []

        return obs


def main():
    env = ENV()
    rx, ry = env.design_reference_line()
    bx1, by1 = env.design_boundary_in()
    bx2, by2 = env.design_boundary_out()

    # plt.plot(rx, ry, linewidth=2, color='darkviolet')
    plt.plot(rx, ry, marker='.')
    plt.plot(bx1, by1, linewidth=1.5, color='k')
    plt.plot(bx2, by2, linewidth=1.5, color='k')
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
