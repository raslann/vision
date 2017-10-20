import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file = sio.loadmat('sfm_points.mat')

image_points = file ['image_points']

centroid = np.mean(image_points, axis= 1, keepdims=True)

image_points -= centroid

W = image_points.transpose(0, 2, 1).reshape(20, 600)

U, D, VT = np.linalg.svd(W)

M = U[:, :3] * D[:3].reshape((1, 3))

world_points = VT.T[:, :3]



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2])

print('The matrix T is:', centroid)
print('********')
print('The matrix M is', M)

plt.show()