import os
import os.path
import csv
from nd2reader import ND2Reader
import pandas as pd

COLUMN_NAMES = ['file name', 'path', 'MB', 'x', 'y', 'z', 'v', 'c']
CSV_PATH = '../data/filenames_DF.csv'
DATA_PATH = '/media/harm/1TB/Cell Data'

df = pd.DataFrame(columns=COLUMN_NAMES)
i = 0
for dirpath, dirname, filenames in os.walk(DATA_PATH):
	for filename in [f for f in filenames if f.endswith('.nd2')]:
		path = os.path.join(dirpath, filename)
		try:
			with ND2Reader(path) as images:
				sizes = images.sizes
				if 'v' not in sizes.keys():
					sizes['v'] = 1
				if 'z' not in sizes.keys():
					sizes['z'] = 1
				if 'c' not in sizes.keys():
					sizes['c'] = 1

		except:
			print("File not readable: {}".format(path))

		# Dont save files with less then 5 images, less then 10 z slices or bad xy
		if sizes['x'] != 256 or sizes['y'] != 256 or \
			 sizes['v'] < 10 or sizes['z'] < 10:
			continue

		file_size = round(os.path.getsize(os.path.join(dirpath, filename)) / 1000000, 2)
		# ['file name', 'path', 'MB', 'x', 'y', 'z', 'v', 'c']
		row = [filename, dirpath[16:], file_size,  sizes['x'], sizes['y'], sizes['z'], sizes['v'], sizes['c']]
		df.loc[i] = row
		i += 1

df.to_csv(CSV_PATH)