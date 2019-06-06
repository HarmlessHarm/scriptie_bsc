import os, os.path, csv
import pandas as pd
import numpy as np

LABEL_FILE = '../data/filenames_DF_labeled.csv'
DATA_PATH = '/media/harm/1TB'
SCT = 'spotcounttable.txt'

df = pd.read_csv(LABEL_FILE, index_col=0)
# df.drop(df.columns[0], axis=1, inplace=True)
df['label_counts'] = ""

for i, row in df.iterrows():
	header = ['0','total', 'inside', 'outside']
	count_file = os.path.join(DATA_PATH, row.label_path, SCT)
	count_data = np.genfromtxt(count_file, delimiter='   ')
	
	# print(count_data[:,1])
	df.at[i,'label_counts'] = list(count_data[:, 1])

df.to_csv(LABEL_FILE)
	# count_df = pd.read_csv(count_file, delimiter='   ', names=header)
	# count_df.drop(count_df.columns.difference(['total']), axis=1, inplace=True)
	# print(count_df.head())