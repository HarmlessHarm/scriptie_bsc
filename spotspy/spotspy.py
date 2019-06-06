from nd2reader import ND2Reader
import numpy as np
# from skimage.feature import blob_log
from math import sqrt
import matplotlib.pyplot as plt
import os
import os.path
from blob import blob_log
from numpy import diff


FILES = [
	{'dir':'2018/MICHAL_DRB TIMECOURSES', 'name':'2018_5_3_AREG_DRBexp_8 0_each001.nd2', 'c': 1},
	{'dir':'2018/MICHAL_DRB TIMECOURSES', 'name':'2018_6_8_DRB24hr_2hrrelease_areg_324each.nd2', 'c': 1},
	{'dir':'2018/MICHAL_DRB TIMECOURSES', 'name':'2018_5_2_PGK1_DRBexp_80_each.nd2', 'c': 1},

]

DATA_DIR = '/media/harm/1TB/Cell Data'

def get_image(file):
	file_path = os.path.join(DATA_DIR, file['dir'], file['name'])
	with ND2Reader(file_path) as images:

		images.bundle_axes = 'zyx'
		images.default_coords['c'] = file['c']
		images.iter_axes = 'v'

		for i, fov in enumerate(images):
			yield fov

def normalize_image(image):
	return image / np.max(image)

def blob_log_3D(norm_image, ms=1, s=5, ns=5, t=[0.1]):
	blobs = blob_log(norm_image, min_sigma=ms, max_sigma=s, num_sigma=ns, thresholds=t)
	# 3D radius sigm * srt(3)
	# if( blobs.shape[1] > 3):
	# 	blobs[:, 3] = blobs[:, 3] * sqrt(3)
	return blobs

# def blob_dog_3D(norm_image, s=5, ns=5, t=0.1):
# 	blobs = glob_dog(norm_image, min_sigma=5, max_sigma=s, num_sigma=ns, threshold=t)
# 	# 3D radius sigm * srt(3)
# 	blobs[:, 3] = blobs[:, 3] * sqrt(3)
# 	return blobs

def write_blobs(file_path, data, i):
	file_name = file_path + str(i).zfill(3) + '.csv'
	import csv
	with open(file_name, 'w') as f:
		blob_writer = csv.writer(f, delimiter=',')
		for blob in data:
			blob_writer.writerow(blob)


file = FILES[2]
file_path = os.path.join(DATA_DIR, file['dir'], file['name'])


image_generator = get_image(file)
import time

colors = ['red', 'blue','green','yellow','cyan','magenta','orange']
for i, image in enumerate(image_generator):
	norm_image = normalize_image(image)

	d_blobs = 0
	p_blobs = 0
	start = time.time()
	prev = start
	# for threshold in np.linspace(0.5,0.05, 10):
		# print(round(t,4))
	ms = 1
	s = 2
	ns = 2
	x = np.linspace(0.55, 0.05, 50)
	log_blobs = blob_log_3D(norm_image, ms=ms, s=s, ns=ns, t=x)
	
	t = time.time() - prev
	# write_blobs('../data/blobs/test_stack_', log_blobs, i)
	# print('Image {}, Thresh: {}. Blobs: {}. Time: {}s'.format(i, round(threshold,4), len(log_blobs), round(t, 2)))
	# new_d_blobs = len(log_blobs) - p_blobs
	# if 
	prev = time.time()
		# print('Image {}. Blobs: {}. Time: {}s'.format(i, len(log_blobs), round(t, 2)))
	# if i == 6:
	# 	break
	print("Image {}, Total elapsed time: {}s".format(i, round(time.time() - start, 2)))
	print("")

	
	y = [x.shape[0] for x in log_blobs]
	dt = 0.5/len(x)	
	dy = np.diff(y) / dt
	dy = list(dy)
	dy.insert(0, dy[0])
	fig, ax1 = plt.subplots()
	ax1.plot(x, y, c='b')
	ax1.set_ylabel('blobs', color='b')
	ax1.tick_params('y', colors='b')
	ax2 = ax1.twinx()
	ax2.plot(x, dy, c='r')
	ax2.set_ylabel('diff', color='r')
	ax2.tick_params('y', colors='r')
	# fig.title("min_sig:{}, max_sig:{}, num_sig:{}".format(ms,s,ns))
	break

plt.show()