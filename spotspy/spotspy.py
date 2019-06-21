from nd2reader import ND2Reader
import numpy as np

# from skimage.feature import blob_log
from math import sqrt
import matplotlib.pyplot as plt
import os, os.path
from blob import blob_log
from numpy import diff
from math import floor, ceil
from scipy.signal import argrelextrema
import pandas as pd
import time
import csv
import multiprocessing as mp
import sys

FILES = [
	{'dir':'2018/MICHAL_DRB TIMECOURSES', 'name':'2018_5_3_AREG_DRBexp_8 0_each001.nd2', 'c': 1},
	{'dir':'2018/MICHAL_DRB TIMECOURSES', 'name':'2018_6_8_DRB24hr_2hrrelease_areg_324each.nd2', 'c': 1},
	{'dir':'2018/MICHAL_DRB TIMECOURSES', 'name':'2018_5_2_PGK1_DRBexp_80_each.nd2', 'c': 1},

]

DATA_DIR = '/media/harm/1TB/'

STORE_KER_9 = list()
STORE_KER_11 = list()
STORE_KER_13 = list()

def get_label_data():
	file = '../data/filenames_DF_labeled.csv'
	labels = pd.read_csv(file, index_col=0)
	good = labels.loc[labels.quality == 'good']
	return good

def get_image(file_path, channel, indices):
	with ND2Reader(file_path) as images:

		images.bundle_axes = 'zyx'
		images.default_coords['c'] = channel
		images.iter_axes = 'v'

		for i, fov in enumerate(images):
			if i in indices:
				print("yield image {}".format(i))
				yield i, fov

def normalize_image(image):
	return image / np.max(image)

def blob_log_3D(norm_image, ms=1, s=2, ns=2, t=[0.1]):
	blobs = blob_log(norm_image, min_sigma=ms, max_sigma=s, num_sigma=ns, thresholds=t)
	return blobs

# def blob_dog_3D(norm_image, s=5, ns=5, t=0.1):
# 	blobs = glob_dog(norm_image, min_sigma=5, max_sigma=s, num_sigma=ns, threshold=t)
# 	# 3D radius sigm * srt(3)
# 	blobs[:, 3] = blobs[:, 3] * sqrt(3)
# 	return blobs

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

def write_blobs(file_path, data, i):
	file_name = file_path + str(i).zfill(3) + '.csv'
	import csv
	with open(file_name, 'w') as f:
		blob_writer = csv.writer(f, delimiter=',')
		for blob in data:
			blob_writer.writerow(blob)


def analyse_image(inpt):
	global img_count, tot_img

	idx, image, ms, s, ns, t, k, plot = inpt

	start = time.time()
	
	norm_image = normalize_image(image)
	
	# params = {'dt': [25], 'kers' : [3,5,7]}
	
	x = np.linspace(0.05, 0.55, t)
	log_blobs = blob_log_3D(norm_image, ms, s, ns, x)

	y = [x.shape[0] for x in log_blobs]

	c = ['y', 'r', 'g']
	# if plot:
	# 	# Plots: Create fig for each param setting and image
	# 	fig, ax1 = plt.subplots()
	# 	fig.suptitle('dt:{}'.format(t))
	# 	ax1.plot(x, y, c='b')
	# 	ax1.scatter(x, [0]*len(x), c='black')
	# 	ax1.set_ylabel('blobs', color='b')
	# 	ax1.tick_params('y', colors='b')

	# 	ax3 = ax1.twinx()
	# 	ax3.set_ylabel('avg deriv.', color='y')
	# 	ax3.tick_params('y', colors='y')

	# 	idx, val, y_plot = find_plateau(k, y, True)
	# 	ax3.plot(x, y_plot, c='y')
	# 	ax1.plot(x[idx], y[idx], 'yo')



		# STORE_KER_9.append(find_plateau(9, y, False))
		# STORE_KER_11.append(find_plateau(11, y, False))
		# STORE_KER_13.append(find_plateau(13, y, False))

	blob_count = find_plateau(k, y, False)
	print("Image {}/{} in {}s".format(img_count.value + 1, tot_img, round(time.time() - start, 2)))
	img_count.value += 1

	return blob_count

def main(at_index):
	labels = get_label_data()

	file_data = labels.loc[at_index]
	file_path = os.path.join(DATA_DIR, file_data.path, file_data.file_name)
	# image_generator = get_image(file_path, file_data.fish_channel)

	series_size = file_data.v
	indices = np.random.choice(np.arange(0,file_data.v), size=50, replace=False)

	global tot_img
	img_count = 0
	tot_img = len(indices)
	DATA = np.empty(series_size)

	# Multiprocessing
	start= time.time()
	p = mp.Pool(4)
	p.map(analyse_image, get_image(file_path, file_data.fish_channel, indices))
	p.close()
	p.join()


	end = round((time.time() - start)/60, 2)
	print("Finished {} images in {}min".format(tot_img, end))

def optimize_params(threads=4):
	global tot_img, img_count
	
	labels = get_label_data()
	params = (1, 2, 2, 25, 5)

	for i, file_data in labels.iterrows():
		
		file_path = os.path.join(DATA_DIR, file_data.path, file_data.file_name)

		# Select 50 random indices from data
		nImages = 10
		img_count = mp.Value('i', 0)
		tot_img = nImages
		indices = np.random.choice(np.arange(0,file_data.v), size=nImages, replace=False)

		img_gen = get_image(file_path, file_data.fish_channel, indices)

		# Multiprocessing
		start= time.time()
		p = mp.Pool(threads)
		test = p.map(analyse_image, [(*img, *params, 1) for i, img in enumerate(img_gen)])
		p.close()
		p.join()
		
		dump(test)

		end = round((time.time() - start)/60, 2)
		print("Finished {} images in {}min".format(tot_img, end))

def dump(data):

	for ()
	print(data)

def parse_args():
	'''
	default arg pattern: [input_path, output_path, params, display/display, multi_threading]
	input_path 		file path or dir path
	output_path		dir_path
	params 				param_file or param_tuple
	plot/display	yes/no
	multi_thread 	number of threads
	'''
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('input', help="Input path (file/dir)", nargs=1, type=str)
	parser.add_argument('-o', dest='output', metavar='OUTPUT FILE/DIR', 
											help="Output path (file/dir)", nargs=1, type=str)
	parser.add_argument('-p', dest='params' ,metavar=('MIN_SIGMA', 'MAX_SIGMA', 
											'NUM_SIGMA', 'NUM_THRESHOLD', 'KERNAL_SIZE'), 
											help="Parameters space separated", nargs=5, type=int)
	parser.add_argument('-pf', dest='param_file', metavar='PARAM FILE', 
											help="Parameters json file", nargs=1, type=str)
	parser.add_argument('-plot', help="Plot yes/no", choices=['yes','no'])
	parser.add_argument('-display', help="Display yes/no",  choices=['yes','no'])
	parser.add_argument('-t', dest='threads', metavar='THREADS', 
											help="Number of threads", nargs=1, type=int)
	
	args = parser.parse_args()

	print(args)
	
	if args.input[0] == "OPTIMIZE":
		print("OPTIMIZE")
		optimize_params()
	

if __name__ == '__main__':
	parse_args()
	# args = sys.argv
	# print(args)

	# labels = get_label_data()
	
	# for i in labels.index:

	# 	main(i)
	# store_data()