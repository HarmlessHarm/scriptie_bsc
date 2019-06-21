import pandas as pd
import numpy as np
import csv



with open('../data/filenames_DF_labeled.csv', 'r') as label_file:
	ldf = pd.read_csv(label_file, index_col=0, header=0)
	for i, row in ldf.iterrows():
		dump_file_name = "dump_multithread_{}.csv".format(i)
		labels = row.label_counts
		with open('../data/dumps/{}'.format(dump_file_name), 'r') as dump_file:
			if row.quality == "good":
				labels = np.fromstring(labels.strip('[]'), dtype=np.float, sep=',').astype(int)
				found = np.fromstring(dump_file.read(), dtype=np.float, sep=',').astype(int)

				diff = np.abs(labels - found)
				print('{}: mean {}, std {}'.format(i, round(np.mean(diff),2), round(np.std(diff),2)))

