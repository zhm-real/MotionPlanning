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
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import HybridAstarPlanner.hybrid_astar as HybridAStar
import HybridAstarPlanner.draw as draw



