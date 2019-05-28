from math import sqrt
from skimage import data
from nd2reader import ND2Reader
import numpy as np
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt


layer = 10
with ND2Reader('test_file.nd2') as images:
	images.bundle_axes = 'czyx'
	frames = images[0][1]
	image = frames[layer]

	print(np.max(image))
	image = image / np.max(image)
	# plt.hist(image)
	# plt.show()

# image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

import time
s = time.time()

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

# print(blobs_log)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
ax = axes.ravel()
for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx+1].set_title(title)
    ax[idx+1].imshow(image, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx+1].add_patch(c)
    ax[idx].set_axis_off()

print('Time: {}'.format(time.time()-s))


plt.tight_layout()

ax[0].set_title("Input layer: {}".format(layer))
ax[0].imshow(image)

plt.show()