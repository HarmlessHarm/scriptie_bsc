import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def make_ax(grid=False):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.grid(grid)
	return ax

def explode(data):
	shape_arr = np.array(data.shape)
	size = shape_arr[:3]*2 - 1
	exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
	exploded[::2, ::2, ::2] = data
	return exploded

def expanded_coordinates(indices):
	x, y, z = indices
	x[1::2, :, :] += 1
	y[:, 1::2, :] += 1
	z[:, :, 1::2] += 1
	return x,y,z

ax = make_ax(True)
colors = np.array([[['#1f77b425']*3]*3]*3)
colors[1,1,1] = '#ff0000ff'
colors = explode(colors)
filled = explode(np.ones((3,3,3)))
x,y,z = expanded_coordinates(np.indices(np.array(filled.shape) + 1))
ax.voxels(x, y, z, filled, facecolors=colors, edgecolors='gray')
plt.show()