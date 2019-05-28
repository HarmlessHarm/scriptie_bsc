from nd2reader import ND2Reader
from skimage.feature import blob_log
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import time


class SlicerViewer(object):
	def __init__(self, ax, X, all_blobs, draw=True):
		self.ax = ax
		self.blobs = all_blobs
		self.draw = draw

		self.X = X
		self.slices, rows, cols = X.shape
		self.ind = self.slices//2

		self.im = self.ax.imshow(self.X[self.ind, :, :])

		self.update()

	def onscroll(self, event):
		if event.button == 'up':
			self.ind = (self.ind + 1)
			if self.ind > self.slices-1: self.ind = self.slices-1
		else:
			self.ind = (self.ind - 1)
			if self.ind < 0: self.ind = 0
		self.update()

	def onclick(self, event):
		if event.key == 'up':
			self.ind = (self.ind + 1)
			if self.ind > self.slices-1: self.ind = self.slices-1
		elif event.key == 'down':
			self.ind = (self.ind - 1)
			if self.ind < 0: self.ind = 0
		self.update()

	def update(self):
		self.im.set_data(self.X[self.ind,:, :])
		self.ax.set_ylabel('slice %s' % self.ind)
		[p.remove() for p in reversed(self.ax.patches)]
		if self.draw:
			for blob in self.blobs[self.ind]:
				y, x, r = blob
				c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
				self.ax.add_patch(c)

		self.im.axes.figure.canvas.draw()

with ND2Reader('data/test_file.nd2') as images:
	images.bundle_axes = 'czyx'
	frames = images[0][1]

s = time.time()

norm_frames = frames / np.max(frames)
blob_count = 0
blobs = list()
for i, frame in enumerate(norm_frames):
	blobs_log = blob_log(frame, max_sigma=5, num_sigma=10, threshold=.1)
	# Compute radii in the 3rd column.
	blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
	blobs.append(blobs_log)
	blob_count += blobs_log.shape[0]

print("Slices time: {}s, blob count: {}".format(round(time.time() - s, 2), blob_count))
s = time.time()
sum_frame = np.sum(frames, axis = 0)
norm_sum_frame = sum_frame / np.max(sum_frame)

sum_blobs_log = blob_log(norm_sum_frame, max_sigma=5, num_sigma=10, threshold=.1)
# Compute radii in the 3rd column.
sum_blobs_log[:, 2] = sum_blobs_log[:, 2] * sqrt(2)
sum_blob_count = sum_blobs_log.shape[0]

print("Summed time: {}s, blob count: {}".format(round(time.time() - s, 2), sum_blob_count))


s = time.time()

blobs_log_3D = blob_log(norm_frames, max_sigma=5, num_sigma=10, threshold=.1)

print("3D time: {}s, blob count: {}".format(round(time.time() - s, 2), blobs_log_3D.shape[0]))

fig, ax = plt.subplots(2, 2)
orig_slices = SlicerViewer(ax[0,0], frames, blobs, False)
draw_slices = SlicerViewer(ax[1,0], frames, blobs)

fig.canvas.mpl_connect('key_press_event', draw_slices.onclick)
fig.canvas.mpl_connect('key_press_event', orig_slices.onclick)


ax[0,1].imshow(sum_frame)
ax[1,1].imshow(sum_frame)

for blob in sum_blobs_log:
	y, x, r = blob
	c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
	ax[1,1].add_patch(c)

plt.show()



# sequence = zip(blobs_list, colors, titles)
# print(*sequence)
# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# ax = axes.ravel()
# for idx, (blobs, color, title) in enumerate(sequence):
#     ax[idx+1].set_title(title)
#     ax[idx+1].imshow(image, interpolation='nearest')
#     for blob in blobs:
#         y, x, r = blob
#         c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#         ax[idx+1].add_patch(c)
#     ax[idx].set_axis_off()

# print('Time: {}'.format(time.time()-s))


# plt.tight_layout()

# ax[0].set_title("Input layer: {}".format(layer))
# ax[0].imshow(image)

# plt.show()