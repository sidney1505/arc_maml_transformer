import os
import json
import numpy as np

def load_data(path='data/training'):
	x_data = []
	y_data = []
	x_val_data = []
	y_val_data = []
	for file in os.listdir(path):
		data_string = open(path + '/' + file, 'r').read()
		data_object = json.loads(data_string)
		#
		x_list = list(map(lambda x: x['input'], data_object['train']))
		y_list = list(map(lambda x: x['output'], data_object['train']))
		x_val_list = list(map(lambda x: x['input'], data_object['test']))
		y_val_list = list(map(lambda x: x['output'], data_object['test']))
		#code.interact(local=dict(globals(), **locals()))
		#
		x = np.zeros([len(x_list),30,30,11])
		y = np.zeros([len(y_list),30,30,11]) # y = np.zeros([len(y_list),30,30], dtype=np.int8)
		x_val = np.zeros([len(x_val_list),30,30,11])
		y_val = np.zeros([len(y_val_list),30,30,11]) # y_val = np.zeros([len(y_val_list),30,30], dtype=np.int8)
		#
		for n in range(x.shape[0]):
			for row_idx in range(x.shape[1]):
				for col_idx in range(x.shape[2]):
					if row_idx < len(x_list[n]) and col_idx < len(x_list[n][row_idx]):
						x[n][row_idx][col_idx][int(x_list[n][row_idx][col_idx])] = 1.0
					else:
						x[n][row_idx][col_idx][10] = 1.0
		#
		for n in range(y.shape[0]):
			for row_idx in range(y.shape[1]):
				for col_idx in range(y.shape[2]):
					if row_idx < len(y_list[n]) and col_idx < len(y_list[n][row_idx]):
						y[n][row_idx][col_idx][int(y_list[n][row_idx][col_idx])] = 1.0 # y[n][row_idx][col_idx] = int(y_list[n][row_idx][col_idx])
					else:
						y[n][row_idx][col_idx][10] = 1.0 # y[n][row_idx][col_idx] = 10
		#
		for n in range(x_val.shape[0]):
			for row_idx in range(x_val.shape[1]):
				for col_idx in range(x_val.shape[2]):
					if row_idx < len(x_val_list[n]) and col_idx < len(x_val_list[n][row_idx]):
						x_val[n][row_idx][col_idx][int(x_val_list[n][row_idx][col_idx])] = 1.0
					else:
						x_val[n][row_idx][col_idx][10] = 1.0
		#
		for n in range(y_val.shape[0]):
			for row_idx in range(y_val.shape[1]):
				for col_idx in range(y_val.shape[2]):
					if row_idx < len(y_val_list[n]) and col_idx < len(y_val_list[n][row_idx]):
						y_val[n][row_idx][col_idx][int(y_val_list[n][row_idx][col_idx])] = 1.0 # y_val[n][row_idx][col_idx] = int(y_val_list[n][row_idx][col_idx])
					else:
						y_val[n][row_idx][col_idx][10] = 1.0# y_val[n][row_idx][col_idx] = 10
		#
		x_data.append(x)
		y_data.append(y)
		x_val_data.append(x_val)
		y_val_data.append(y_val)
	return {'x' : x_data, 'x_val' : x_val_data, 'y' : y_data, 'y_val' : y_val_data}
