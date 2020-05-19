import numpy as np

def accuracy(model, x, y, config):
	y_pred = model.predict(x)
	# y_pred = np.argmax(y_pred,axis=-1)
	y_pred_int = np.argmax(y_pred,axis=-1)
	y_int = np.argmax(y,axis=-1)
	number_correct = 0
	total_correct = 0
	num_invalid = 0
	number_valid_samples = 0
	#
	true_class_distribution = np.zeros(11)
	pred_class_distribution = np.zeros(11)
	for i in range(y_pred_int.shape[0]):
		total = True
		invalid = False
		for j in range(y_pred_int.shape[1]):
			for k in range(y_pred_int.shape[2]):
				if y_int[i][j][k] != 10:
					true_class_distribution[y_int[i][j][k]] += 1.0
					pred_class_distribution[y_pred_int[i][j][k]] += 1.0
					number_valid_samples += 1
					if y_pred_int[i][j][k] == y_int[i][j][k]:
						number_correct += 1
					else:
						total = False
					if y_pred_int[i][j][k] == 10:
						num_invalid += 1
		if total:
			total_correct += 1
	ratio_correct = number_correct / number_valid_samples
	ratio_total = total_correct / y_pred_int.shape[0]
	ratio_invalid = num_invalid / number_valid_samples
	true_class_distribution = true_class_distribution / np.sum(true_class_distribution)
	pred_class_distribution = pred_class_distribution / np.sum(pred_class_distribution)
	if config['verbosity'] >= 2:
		print('ratio_correct: ' + str(ratio_correct))
		print(list(true_class_distribution))
		print(list(pred_class_distribution))
	return ratio_correct, ratio_total, ratio_invalid
