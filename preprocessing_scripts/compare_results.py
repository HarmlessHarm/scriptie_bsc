import numpy as np
import pickle
import os
import pprint
import math
# from sklearn.model_selection import ParameterGrid
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


dir_path = os.path.dirname(os.path.realpath(__file__))
pkl_dir = os.path.abspath(os.path.join(dir_path, '../data/pkls'))


# MS = np.arange(2,7,2)
# NS = np.arange(2,7,2)
# NT = np.arange(20,41,10)
# K = np.arange(2,15,2)

# P = [[1], MS, NS, NT, K]

params_dict = defaultdict(lambda: defaultdict(list))

# all_labels = np.empty(0,int)
all_labels = list()

plt_mae = []
plt_nmae = []

err = 'nmae'

best_parameters = []

for file in os.listdir(pkl_dir):
	file_path = os.path.join(pkl_dir, file)

	with open(file_path, 'rb') as pkl:
		data = pickle.load(pkl)

		best_err = float('inf')
		errors = list()

		# all_labels = np.concatenate((all_labels, data['analyzed_labels']))
		all_labels += list(data['analyzed_labels'])

		xdata = 0
		for i, ana in enumerate(data['analyses']):
			# Save all configurations in big dict
			params = ana['params']
			params_dict[params]['counts'] += list(ana['counts'])
			mean = np.mean(ana['counts'])
			params_dict[params]['norm_err'] += [round(ana['mae'],2) / mean]
			params_dict[params]['std'] += [round(ana['std'],2) / mean]
			params_dict[params]['time'] += [round(ana['time'],2)]
			
			mean = np.mean(ana['counts'])
			errors.append((params, round(ana[err], 2)))

			plt_mae.append(ana['mae'])
			plt_nmae.append(ana['nmae'])
			xdata += 1

		for pars, e in sorted(errors, key=lambda x:x[1])[:10]:
			# if pars not in best_parameters:
			best_parameters.append(pars)
			


def plot_prediction_scatter():

	best_mae = float('inf')
	best_rmse = float('inf')
	best_nmae = float('inf')

	best_pears = -1
	pears_equiv = list()

	for pars in params_dict:
		counts = params_dict[pars]['counts']
		diff = (abs(np.array(counts) - np.array(all_labels)))
		mae = np.mean(diff)
		mse = np.mean(diff ** 2)
		rmse = math.sqrt(mse)

		from scipy import stats

		pears = stats.pearsonr(counts, all_labels)[0]
		nmae = np.mean(params_dict[pars]['norm_err'])

		params_dict[pars]['total_mae'] = mae
		params_dict[pars]['total_mse'] = mse
		params_dict[pars]['total_rmse'] = rmse
		params_dict[pars]['pears'] = pears
		params_dict[pars]['total_nmae'] = nmae

		if mae < best_mae:
			par_mae = pars
			best_mae = mae

		if nmae < best_nmae:
			par_nmae = pars
			best_nmae = nmae

		if rmse < best_rmse:
			par_rmse = pars
			best_rmse = rmse

		if pears > best_pears:
			par_pears = pars
			best_pears = pears
			pears_equiv = list()
		
		if pears == best_pears:
			pears_equiv.append(pars)

	print(par_nmae, best_nmae)

	# Pearsons
	# plot_param = par_pears
	# Histogram
	# plot_param = (1,2,6,20,6)
	# General
	print(par_pears, np.mean(params_dict[par_pears]['norm_err']))

	plot_param = (1,2,2,40,2)
	print(plot_param, np.mean(params_dict[plot_param]['norm_err']))
	plot_param = (1,2,2,30,2)
	print(plot_param, np.mean(params_dict[plot_param]['norm_err']))




	x = params_dict[plot_param]['counts']
	# y = params_dict[(1,2,2,20,2)]['counts']
	y = all_labels

	for i in range(14):
		if i == 4:
			continue
		i,j = (i*50, (i+1) * 50)
		plt.scatter(x[i:j], y[i:j], alpha=0.2)

	# plt.xlim(-1,100)
	# plt.ylim(-1,100)
	plt.title("Parameter-set: {}".format(plot_param), fontsize=14)
	plt.xlabel('Predicted number of spots', fontsize=14)
	plt.ylabel('Real number of spots', fontsize=14)
	plt.show()

# print(par_mae, best_mae)
# print(par_rmse, best_rmse)

# def plot_comparison():


col_14 = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','b','r','y','g']

def plot_spot_counts():
	bins1 = np.arange(0, 10)
	bins10 = np.arange(10, 100, 10)
	bins100 = np.arange(100, 900, 25)

	vals = [all_labels[i*50:(i + 1)*50] for i in range(14)]

	fig, (ax1, ax2, ax3) = plt.subplots(3,1)
	# fig.suptitle("Labelled spot counts")

	# ax1.set_color_cycle(['red', 'black', 'yellow'])
	ax1.set(xlabel='spot count range 0-10')
	ax2.set(xlabel='spot count range 10-100')
	ax3.set(xlabel='spot count range 100-900')


	ax1.hist(vals, bins1, histtype='barstacked', color=col_14)
	ax2.hist(vals, bins10, histtype='barstacked', color=col_14)
	ax3.hist(vals, bins100, histtype='barstacked', color=col_14)


	plt.show()


def plot_parameter_hist():

	sigs_x = [2,4,6]
	thresh_x = [20,30,40]
	kers_x = [2,4,6,8,10,12,14]


	max_sigs = dict().fromkeys(sigs_x, 0)
	num_sigs = dict().fromkeys(sigs_x, 0)
	thresh = dict().fromkeys(thresh_x, 0)
	kers = dict().fromkeys(kers_x, 0)

	for _, ms, ns, t, k in best_parameters:
		max_sigs[ms] += 1
		num_sigs[ns] += 1
		thresh[t] += 1
		kers[k] += 1

	bar_max_sigs = list(zip(*sorted(zip(max_sigs.keys(), max_sigs.values()), key=lambda x:x[0])))
	bar_num_sigs = list(zip(*sorted(zip(num_sigs.keys(), num_sigs.values()), key=lambda x:x[0])))
	bar_thresh = list(zip(*sorted(zip(thresh.keys(), thresh.values()), key=lambda x:x[0])))
	bar_kers = list(zip(*sorted(zip(kers.keys(), kers.values()), key=lambda x:x[0])))

	row, col = 2,2
	fig, axes = plt.subplots(row, col)
	fig.suptitle("Parameter frequency in top 10 parameter set")

	axes[0,0].bar(range(len(sigs_x)), bar_max_sigs[1], tick_label=sigs_x)
	axes[0,1].bar(range(len(sigs_x)), bar_num_sigs[1], tick_label=sigs_x)
	axes[1,0].bar(range(len(thresh_x)), bar_thresh[1], tick_label=thresh_x)
	axes[1,1].bar(range(len(kers_x)), bar_kers[1], tick_label=kers_x)
	
	axes[0,0].set_title("Max sigma")
	axes[0,1].set_title("Number of sigmas")
	axes[1,0].set_title("Number of thresholds")
	axes[1,1].set_title("Kernel size")

	plt.show()


def plot_parametersets():
	# Plot of best parameters
	# MODIFY THE NUMBER OF BEST PARAMETERS!!!
	x = list(range(1,15))

	colors = ['green', 'gold', 'orange', 'red', 'darkred']
	# row, col = 4, 4
	# fig, axes = plt.subplots(row, col, sharex=True, sharey=True)
	# fig.suptitle('Mean Absolute Errors of parameter sets')


	rows = len(best_parameters)
	cols = 14
	heatmap_array = np.empty((rows, cols)) 
	print(heatmap_array.shape)
	for i, pars in enumerate(best_parameters):
		heatmap_array[i, :] = np.array(params_dict[pars]['norm_err']).reshape(1,14)


	heatmap_bitmap = np.clip(heatmap_array, 0, 2.5)
	# heatmap_bitmap = heatmap_array
	heatmap_bitmap *= -1
	# heatmap_bitmap = (heatmap_array <= np.max(heatmap_array.diagonal()))

	heatmap_bitmap = np.zeros((14,14))
	np.fill_diagonal(heatmap_bitmap, 1)
	heatmap_bitmap[9,:] = np.ones((1,14))


	fig, ax = plt.subplots()
	
	im = ax.imshow(heatmap_bitmap, cmap='RdYlGn')

	ax.set_xticks(np.arange(cols))
	ax.set_yticks(np.arange(rows))

	import string
	letters = list(string.ascii_lowercase)[:14]
	
	ax.set_xticklabels(np.arange(cols) +1)
	ax.set_yticklabels(letters)

	ax.set_xlabel('Image sets', fontsize=18)
	ax.set_ylabel('Parameter sets', fontsize=18)

	# for i in range(rows):
	# 	for j in range(cols):
	# 		val = round(-1 * heatmap_bitmap[i,j],2)
	# 		if val >= 2.5:
	# 			val = "2.5+"
	# 		text = ax.text(j, i, val, 
	# 			ha='center', va='center', color='w')

	# plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
	plt.tight_layout()
	plt.show()


def plot_mse_dist():
	# 
	end = len(plt_mae)

	fig, ax1 = plt.subplots()
	fig.suptitle('Mean Absolute Errors of parameter sets per image set')

	ax1.set_ylabel('MAE')
	# ax1.tick_params('y')
	
	ax1.set_xlabel('Image file')
	ax1.set_xlim(0,end)
	ax1.set_ylim(0,175)
	ticks = np.arange(0,end,xdata) + (xdata / 2)
	ax1.set_xticks(ticks)
	ax1.set_xticklabels(np.arange(1,len(ticks) + 1))


	ax2 = ax1.twinx()
	ax2.set_ylabel('Normalized MAE', color='orange')
	ax2.set_ylim(0,30)
	ax2.tick_params(axis='y', color='orange')
	ax2.spines['right'].set_color('orange')
	# ax2.xaxis.label.set_color('orange')
	# ax2.set_ylabel('')
	plt.setp(ax2.get_yticklabels(), color='orange')

	x = list(range(end))
	ax1.scatter(x, plt_mae, marker='.', s=5)
	ax2.scatter(x, plt_nmae, c='orange', marker='.', s=5)

	for x in range(0, end, xdata):
		ax1.axvline(linewidth=1, color='black', x=x)

	# ax1.set_ylim(0,1000)
	plt.show()


# plot_prediction_scatter()
# plot_parametersets()
# plot_parameter_hist()
plot_spot_counts()
# plot_mse_dist()