import threading
from nd2reader import ND2Reader
import multiprocessing.dummy as mp
import pandas as pd
from blob import blob_log
import numpy as np
from scipy.signal import argrelextrema
import time, os, os.path
from math import ceil, floor
import csv

DATA_DIR = '/media/harm/1TB/'


def get_image(file_path, channel):
	with ND2Reader(file_path) as images:
		images.bundle_axes = 'zyx'
		images.default_coords['c'] = channel - 1
		images.iter_axes = 'v'

		for i, fov in enumerate(images):
			yield i, fov

def get_label_data():
	file = '../data/filenames_DF_labeled.csv'
	labels = pd.read_csv(file, index_col=0)
	return labels

def blob_log_3D(norm_image, ms=1, s=2, ns=2, t=[0.1]):
	blobs = blob_log(norm_image, min_sigma=ms, max_sigma=s, num_sigma=ns, thresholds=t)
	return blobs

def find_plateau(ker_size, y, plot=False):
	conv_ker = [-0.5] + [0]*(ker_size - 2) + [0.5]
	p = (ker_size-1)/2
	pad = (floor(p), ceil(p))
	y_pad = np.pad(y, pad, 'edge')
	y_conv = np.convolve(y_pad, conv_ker, 'valid')
	# some minima have plateau, wrap to exclude begin dips
	min_i = argrelextrema(y_conv, np.less_equal, order=2, mode='wrap')[0]

	# first plateau is 
	plat_val = np.array(y)[min_i][0]
	plat_idx = min_i[0]
	# print("ker size: {}, blobs: {}".format(ker_size, plat_val))
	if plot:
		return plat_idx, plat_val, y_conv
	return plat_val

def normalize_image(image):
	return image / np.max(image)

def analyse_image(inpt):
	global DATA
	i, image = inpt
	s = time.time()
	norm_image = normalize_image(image)
	x = np.linspace(0.05, 0.55, 25)
	log_blobs = blob_log_3D(norm_image, ms=1, s=2, ns=2, t=x)
	y = [x.shape[0] for x in log_blobs]
	blob_count = find_plateau(5, y, False)
	DATA[i] = blob_count
	print("Image {} found {} blobs in {}s".format(i, blob_count, round(time.time() - s, 2)))

def main(idx):
	global DATA
	labels = get_label_data()

	file_data = labels.loc[idx]
	file_path = os.path.join(DATA_DIR, file_data.path, file_data.file_name)
	data_size = file_data.v
	DATA = np.empty(data_size)
	# image_generator = get_image(file_path, file_data.fish_channel)
	start = time.time()
	p = mp.Pool(4)
	p.map(analyse_image, get_image(file_path, file_data.fish_channel))
	p.close()
	p.join()
	print("finished in {}s".format(round(time.time() - start, 2)))


	with open('dump_multithread_{}.csv'.format(idx), 'w') as file:
		csv_writer = csv.writer(file, delimiter=',')
		csv_writer.writerow(DATA)


if __name__ == '__main__':
	import sys
	idx = 53

	if len(sys.argv) > 1:
		idx = int(sys.argv[1])
	main(idx)