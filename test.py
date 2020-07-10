import scipy.spatial.kdtree as KD

obs = [[0, 0], [0, 1], [0, 2], [0, 3]]

kdtree = KD.KDTree(obs)

points = kdtree.query_ball_point([1, 1], 1.5)

print(points)
