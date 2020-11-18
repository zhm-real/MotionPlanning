## Overview
This repository implemented some common motion planners used on autonomous vehicles, including
* [Hybrid A* Planner](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf)
* [Frenet Optimal Trajectory](https://www.researchgate.net/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame)
* [Hierarchical Optimization-Based Collision Avoidance (H-OBCA)](https://ieeexplore.ieee.org/document/9062306) (Incomplete)

Also, this repository provides some controllers for path tracking, including
* [Pure Pursuit + PID](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf)
* [Rear-Wheel Feedback + PID](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
* [Front-Wheel Feedback / Stanley + PID](http://robots.stanford.edu/papers/thrun.stanley05.pdf)
* [LQR + PID](https://github.com/ApolloAuto/apollo/tree/master/modules/control/controller)
* [Linear MPC](https://borrelli.me.berkeley.edu/pdfpub/pub-6.pdf)

## Requirement
* Python 3.6 or above
* [SciPy](https://www.scipy.org/)
* [cvxpy](https://github.com/cvxgrp/cvxpy)
* [Reeds-Shepp Curves](https://github.com/zhm-real/ReedsSheppCurves)
* [pycubicspline](https://github.com/AtsushiSakai/pycubicspline)

## Vehicle models
This repository uses two models: simple car model and [car pulling trailers model](http://planning.cs.uiuc.edu/node661.html#77556).

## Hybrid A* Planner
<div align=right>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/HybridAstarPlanner/gif/hybrid%20Astar-1.gif" alt="1" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/HybridAstarPlanner/gif/hybrid%20Astar-2.gif" alt="2" width="400"/></a></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/HybridAstarPlanner/gif/hybrid%20Astar-t1.gif" alt="11" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/HybridAstarPlanner/gif/hybrid%20Astar-t5.gif" alt="12" width="400"/></a></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/HybridAstarPlanner/gif/hybrid%20Astar-t3.gif" alt="13" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/HybridAstarPlanner/gif/hybrid%20Astar-t2.gif" alt="14" width="400"/></a></td>
  </tr>
</table>
</div>

## State Lattice Planner
<div align=right>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/LatticePlanner/gif/Crusing.gif" alt="1" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/LatticePlanner/gif/StoppingMode.gif" alt="2" width="400"/></a></td>
  </tr>
</table>
</div>

## Controllers
<div align=right>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/Control/gif/purepursuit1.gif" alt="1" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/Control/gif/RWF1.gif" alt="2" width="400"/></a></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/Control/gif/stanley.gif" alt="1" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/Control/gif/MPC.gif" alt="2" width="400"/></a></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/Control/gif/LQR_Kinematic.gif" alt="1" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/Control/gif/LQR_Dynamics.gif" alt="2" width="400"/></a></td>
  </tr>
</table>
</div>

## Paper
### Planning
* [Basic Path Planning Algorithms: ](https://github.com/zhm-real/PathPlanning) PathPlanning
* [Baidu Apollo Planning module: ](https://github.com/ApolloAuto/apollo/tree/master/modules/planning) Recommended Materials
* [Survey of Planning and Control algos: ](https://arxiv.org/pdf/1604.07446.pdf) A Survey of Motion Planning and Control Techniques for Self-driving Urban Vehicles
* [Hybrid A* Planner: ](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf) Practical Search Techniques in Path Planning for Autonomous Driving
* [Frenet Optimal Trajectory: ](https://www.researchgate.net/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame) Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame

### Control
* [Baidu Apollo Control module: ](https://github.com/ApolloAuto/apollo/tree/master/modules/control) Recommended Materials
* [Pure Pursuit: ](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf) Implementation of the Pure Pursuit Path Tracking Algorithm 
* [Rear-Wheel Feedback: ](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf) Automatic Steering Methods for Autonomous Automobile Path Tracking
* [Front-Wheel Feedback / Stanley: ](http://robots.stanford.edu/papers/thrun.stanley05.pdf) Stanley: The Robot that Won the DARPA Grand Challenge
* [LQR: ](https://github.com/ApolloAuto/apollo/tree/master/modules/control/controller) ApolloAuto/apollo: An open autonomous driving platform
* [Linear MPC: ](https://borrelli.me.berkeley.edu/pdfpub/pub-6.pdf) MPC-Based Approach to Active Steering for Autonomous Vehicle Systems

## Useful Material
* [HybridAStarTrailer (Julia): ](https://github.com/AtsushiSakai/HybridAStarTrailer) by AtsushiSakai
* [Hybrid Path Planner (C++): ](https://github.com/karlkurzer/path_planner) by KTH Research Concept Vehicle
