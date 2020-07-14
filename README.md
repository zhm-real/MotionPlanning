# HybridAstar
## Overview
This is a python version of Hybrid A* algorithm, which can generate smooth paths for an autonomous vehicle operating in an unknown environment with obstacles. This algorithm was proposed by Stanford and was experimentally validated in 2007 DARPA Urban Challenge.

## Requirement
* Python3
* [SciPy](https://www.scipy.org/)
* [Reeds-Shepp Curves](https://github.com/zhm-real/ReedsSheppCurves)

## Vehicle models
This repository uses two models: simple car model and [car pulling trailers model](http://planning.cs.uiuc.edu/node661.html#77556).

## Path Planning Simulation
<div align=right>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/HybridAstar/blob/master/gif/hybrid%20Astar-1.gif" alt="1" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/HybridAstar/blob/master/gif/hybrid%20Astar-2.gif" alt="2" width="400"/></a></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/HybridAstar/blob/master/gif/hybrid%20Astar-t1.gif" alt="11" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/HybridAstar/blob/master/gif/hybrid%20Astar-t1.gif" alt="12" width="400"/></a></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="https://github.com/zhm-real/HybridAstar/blob/master/gif/hybrid%20Astar-t2.gif" alt="11" width="400"/></a></td>
    <td><img src="https://github.com/zhm-real/HybridAstar/blob/master/gif/hybrid%20Astar-t3.gif" alt="12" width="400"/></a></td>
  </tr>
</table>
</div>

## Useful Material
* [Practical Search Techniques in Path Planning for Autonomous Driving](https://ai.stanford.edu/~ddolgov/papers/dolgov_gpp_stair08.pdf) by Stanford
* [Hybrid Path Planner (C++)](https://github.com/karlkurzer/path_planner) by KTH Research Concept Vehicle
* [hybrid-astar-planner (MATLAB)](https://github.com/wanghuohuo0716/hybrid_A_star) by Mengli Liu
* [HybridAStarTrailer (Julia)](https://github.com/AtsushiSakai/HybridAStarTrailer) by AtsushiSakai
