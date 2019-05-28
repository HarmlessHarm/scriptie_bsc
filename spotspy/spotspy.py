from nd2reader import ND2Reader
import numpy as np
from skimage.feature import blob_log
from math import sqrt

files = [
	{'name':'test_file.nd2', 'c': 1},
	{'name':'test_stack.nd2', 'c': 1},
]

data_dir = '../data/'

def get_image(file):
	file_path = data_dir + file['name']
	with ND2Reader(file_path) as images:

		images.bundle_axes = 'zyx'
		images.default_coords['c'] = file['c']
		images.iter_axes = 'v'

		for i, fov in enumerate(images):
			yield fov

def normalize_image(image):
	return image / np.max(image)

def blob_log_3D(image):
	blobs = blob_log(normalize_image(image), max_sigma=5, num_sigma=10, threshold=0.1)
	# 3D radius srt(3) * sigm
	blobs[:, 3] = blobs[:, 3] * sqrt(3)
	return blobs

def write_blobs(file_path, data, i):
	file_name = file_path + str(i).zfill(3) + '.csv'
	import csv
	with open(file_name, 'w') as f:
		blob_writer = csv.writer(f, delimiter=',')
		for blob in data:
			blob_writer.writerow(blob)
		


file_path = data_dir+files[1]['name']
image_generator = get_image(files[1])

import time
start = time.time()
prev = start
for i, image in enumerate(image_generator):
	blobs = blob_log_3D(image)
	t = time.time() - prev
	write_blobs('../data/blobs/test_stack_', blobs, i)
	prev = time.time()
	print('Image {}. Blobs: {}. Time: {}s'.format(i, len(blobs), round(t, 2)))

print("Total elapsed time: {}s".format(round(time.time() - start, 2)))