import scipy.spatial.kdtree as KD
import numpy as np

obs = [[0, 0], [0, 1], [0, 2], [0, 3]]

points = np.array(obs)

kdtree = KD.KDTree(points)

point = kdtree.query_ball_point([1, 1], 1.5)

print(point)
