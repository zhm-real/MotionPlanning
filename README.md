# Motion Planner
## Overview
This repository implemented some common motion planners in autonomous vehicles, including Hybrid A*, Lattice Planner and EM Planner (imcompleted). Also, this repository provides some controllers such as purpure pursuit, MPC, to track the path from planners.

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
    <td><img src="https://github.com/zhm-real/MotionPlanning/blob/master/LatticePlanner/gif/Stopping.gif" alt="2" width="400"/></a></td>
  </tr>
</table>
</div>

## Useful Material
* [Practical Search Techniques in Path Planning for Autonomous Driving](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf) by Stanford
* [Hybrid Path Planner (C++)](https://github.com/karlkurzer/path_planner) by KTH Research Concept Vehicle
* [hybrid-astar-planner (MATLAB)](https://github.com/wanghuohuo0716/hybrid_A_star) by Mengli Liu
* [HybridAStarTrailer (Julia)](https://github.com/AtsushiSakai/HybridAStarTrailer) by AtsushiSakai
