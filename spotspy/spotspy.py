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
import pickle

DATA_DIR = '/media/harm/1TB/'

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

def generate_plot(x, y, y2, t):
	# Plots: Create fig for each param setting and image
	fig, ax1 = plt.subplots()
	fig.suptitle('dt:{}'.format(t))
	ax1.plot(x, y, c='b')
	ax1.scatter(x, [0]*len(x), c='black')
	ax1.set_ylabel('blobs', color='b')
	ax1.tick_params('y', colors='b')

	ax3 = ax1.twinx()
	ax3.set_ylabel('avg deriv.', color='y')
	ax3.tick_params('y', colors='y')

	ax3.plot(x, y2, c='y')
	ax1.plot(x[t], y[t], 'yo')

	plt.show()

def generate_display():
	pass

def analyse_image(inpt):
	idx, image, ms, s, ns, t, ks, plot = inpt

	start = time.time()
	
	norm_image = normalize_image(image)
	
	x	= np.linspace(0.05, 0.55, t)
	log_blobs = blob_log_3D(norm_image, ms, s, ns, x)

	y = [x.shape[0] for x in log_blobs]

	analysis = list()
	for k in ks:
		blob_count = find_plateau(k, y, False)
		analysis.append(blob_count)
	
	print("Image {} in {}s".format(idx, round(time.time() - start, 2)), end='\r')
	# print('.', end='')
	# img_count.value += 1

	return analysis

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

def optimize_params(threads=6):
	global tot_img, img_count
	
	labels = get_label_data()
	params = (1, 2, 2, 25, 5)
	ms = 1
	ss = range(2,7,2)
	nss = range(2,7,2)
	ts = [20, 30, 40]
	ks = range(2,15,2)

	opt_start = time.time()

	for i, file_data in labels.iterrows():
		
		if i in [27,36,46,48,53,56,58,62]:
			continue

		file_path = os.path.join(DATA_DIR, file_data.path, file_data.file_name)

		# Select 50 random indices from data
		nImages = 50
		img_count = mp.Value('i', 0)
		tot_img = nImages
		indices = np.random.choice(np.arange(0,file_data.v), size=nImages, replace=False)


		data_dict = dict()
		data_dict['meta_data'] = file_data.to_dict()
		data_dict['meta_data']['label_counts'] = np.array(list(map(int, map(float, data_dict['meta_data']['label_counts'].strip('[]').split(',')))))
		data_dict['indices'] = indices
		data_dict['analyzed_labels'] = data_dict['meta_data']['label_counts'][indices]

		data_dict['analyses'] = list()

		print('Optimizing {}:{} for {} parameter combinations'.format(i, file_data.file_name, len(ts) * len(ss) * len(nss)))
		combi = 1
		for t in ts:
			for s in ss:
				for ns in nss:
					params = (ms, s, ns, t, ks)
					# print("Starting analysis {}".format(nImages))
					start= time.time()
					# print("Params: {},{},{},{}".format(*params[:-1]))
					img_gen = get_image(file_path, file_data.fish_channel, indices)
					# Multiprocessing
					p = mp.Pool(threads)
					analysis = p.map(analyse_image, [(idx, img[1], *params, 1) for idx, img in enumerate(img_gen)])
					
					arr = np.asarray(analysis)

					p.close()
					p.join()

					end_sec = round(time.time() - start, 2)
					end_min = round((time.time() - start) / 60, 2)

					for i, col in enumerate(arr.T):
						cnts = data_dict['analyzed_labels']
						diff = abs(cnts - col)
						diff2 = (cnts - col) ** 2
						mean = np.mean(diff)
						std = np.std(diff)
						mean2 = np.mean(diff2)
						std2 = np.std(diff2)
						data_dict['analyses'] += [{
							'params':(ms, s, ns, t, ks[i]), 
							'counts': col, 
							'mean':mean,
							'std':std,
							'sq_mean':mean2,
							'sq_std':std2,
							'time': end_sec
						}]


					print("{}: Finished with params: S:{}, nS:{}, nT:{} in {}min".format(combi, *params[1:-1], end_min))
					combi += 1
		dump(data_dict)
		print("Finished data set {} in: {}s".format(file_data.file_name, round(time.time() - opt_start, 2)))

def dump(data):

	file_name = data['meta_data']['file_name']
	file_name = file_name[:-3] + 'pkl'
	print(file_name)
	with open('../data/pkls/' + file_name, 'wb') as dump_file:
		pickle.dump(data, dump_file)

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
	
	if args.input[0] == "OPTIMIZE":
		optimize_params(4)
	

if __name__ == '__main__':
	parse_args()
	# args = sys.argv
	# print(args)

	# labels = get_label_data()
	
	# for i in labels.index:

	# 	main(i)
	# store_data()