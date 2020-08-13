"""
velocity smoother: smooth velocity
"""

import numpy as np


def veloSmooth(v, amax, Ts):
    l_v = len(v)
    v_ex = np.zeros(l_v + 40)
    v_bar = np.zeros((4, l_v + 40))
    v_bar2 = np.zeros((4, l_v + 40))
    v_barMM = np.zeros((1, l_v))

    for i in range(l_v):
        v_ex[i + 19] = v[i]

        for j in range(4):
            v_bar[j][i + 19] = v[i]
            v_ex[i + 19] = v[i]

    v_cut1 = 0.25 * abs(v[0])
    v_cut2 = 0.25 * abs(v[0]) + abs(v[0])

    accPhase = int(round(abs(v[0]) / amax / Ts))

    v_ex_diff = np.diff(v_ex)

    index1 = [ind for ind in range(l_v - 1) if v_cut1 < v_ex_diff[ind] < v_cut2]
    index2 = [ind for ind in range(l_v - 1) if v_ex_diff[ind] > v_cut2]

    index3 = [ind for ind in range(l_v - 1) if -v_cut2 < v_ex_diff[ind] < -v_cut1]
    index4 = [ind for ind in range(l_v - 1) if v_ex_diff[ind] < -v_cut2]

    if len(index1) >= 1 and index1[0] == 18:
        index1[0] += 1

    if len(index3) >= 1 and index3[0] == 18:
        index3[0] += 1

    for j in range(len(index1)):
        if v_ex[index1[j]] > v_cut1 or v_ex[index1[j] + 1] > v_cut1:
            v_bar[0][index1[j]: index1[j] + accPhase] = np.arange(0, abs(v[0]), accPhase + 1)
        elif v_ex[index1[j]] < -v_cut1 or v_ex[index1[j] + 1] < -v_cut1:
            v_bar[0][index1[j] - accPhase + 1: index1[j] + 1] = np.arange(-abs(v[0]), 0, accPhase + 1)

    for j in range(len(index3)):
        if v_ex[index3[j]] > v_cut1 or v_ex[index3[j] + 1] > v_cut1:
            return
