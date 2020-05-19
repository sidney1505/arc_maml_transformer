import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
import datetime
import os
import json
import random
import code  # code.interact(local=dict(globals(), **locals()))

# imports from keras_transformer, which was adapted in order to fit the purpose
from keras_transformer.attention import MultiHeadSelfAttention
from keras_transformer.position import TransformerCoordinateEmbedding, AddPositionalEncoding
from keras_transformer.transformer import TransformerACT, TransformerBlock

# imports from own libraries
from losses import categorical_crossentropy, categorical_crossentropy_masked, focal_loss, focal_loss_masked
from metrics import accuracy

class ARCModel:
	def __init__(self, config):
		#
		self.config = config
		#
		if not self.config['is_loaded']:
			self.config['name'] = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
			os.makedirs('saved_models/' + self.config['name'])
			os.makedirs('saved_models/' + self.config['name'] + '/weights')
			os.makedirs('saved_models/' + self.config['name'] + '/logs')
		#
		self.loss = categorical_crossentropy
		#
		self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.0)
		#
		self.model = tf.keras.Sequential()
		if self.config['model_type'] == 'conv':
			size = 4
			self.model.add(tf.keras.layers.Conv2D(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 32, 3, activation='relu', padding='same', strides=(1, 2)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			#
			self.model.add(tf.keras.layers.Conv2D(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 64, 3, activation='relu', padding='same', strides=(2, 1)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			# transformer layer
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 64, 3, activation='relu', padding='same', strides=(2, 1)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			#
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 32, 3, activation='relu', padding='same', strides=(1, 2)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			#
			self.model.add(tf.keras.layers.Conv2DTranspose(11, 1, activation='softmax', padding='same'))
		elif self.config['model_type'] == 'transformer':
			size = 4
			self.model.add(tf.keras.layers.Conv2D(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 32, 3, activation='relu', padding='same', strides=(1, 2)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			#
			self.model.add(tf.keras.layers.Conv2D(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2D(size * 64, 3, activation='relu', padding='same', strides=(2, 1)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			# transformer layer
			self.model.add(tf.keras.layers.Reshape([15 * 15, size * 64]))
			#
			self.model.add(TransformerBlock('Transformer1', num_heads=8, use_masking=False))
			#self.model.add(TransformerBlock('Transformer2', num_heads=8, use_masking=False))
			#self.model.add(TransformerBlock('Transformer3', num_heads=8, use_masking=False))
			#self.model.add(TransformerBlock('Transformer4', num_heads=8, use_masking=False))
			self.model.add(tf.keras.layers.Reshape([15, 15, size * 64]))
			#
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 64, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 64, 3, activation='relu', padding='same', strides=(2, 1)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			#
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 32, 3, activation='relu', padding='same'))
			self.model.add(tf.keras.layers.Conv2DTranspose(size * 32, 3, activation='relu', padding='same', strides=(1, 2)))
			self.model.add(tf.keras.layers.LayerNormalization(trainable=True))
			#
			self.model.add(tf.keras.layers.Conv2DTranspose(11, 1, activation='softmax', padding='same'))
		elif self.config['model_type'] == 'fc':
			self.model = tf.keras.Sequential()
			self.model.add(tf.keras.layers.Reshape([30 * 30 * 11]))
			self.model.add(tf.keras.layers.Dense(30 * 30 * 11, activation='tanh'))
			self.model.add(tf.keras.layers.Dense(30 * 30 * 11))
			self.model.add(tf.keras.layers.Reshape([30, 30, 11]))
			self.model.add(tf.keras.layers.Activation(tf.keras.activations.softmax))
		#
		self.model.compile(optimizer=self.optimizer, loss=categorical_crossentropy)
		self.model.build(input_shape=[None,30,30,11])
		#
		if self.config['is_loaded']:
			self.model.load_weights('saved_models/' + self.config['name'] + '/weights/' + str(self.config['current_epoch']))
		else:
			open('saved_models/' + self.config['name'] + '/config.json', 'w').write(json.dumps(self.config, sort_keys=True, indent=4))
			self.model.save_weights('saved_models/' + self.config['name'] + '/weights/' + str(self.config['current_epoch']))

	# e.g. train_on_one_example(train_data['x'][48], train_data['y'][48])
	def train_on_one_example(self, x, y):
		# self.model.fit(x, y, epochs=1000, verbose=1)
		learning_rate = self.config['learning_rate']
		ratio_correct = None
		for i in range(1000):
			print(i)
			if i % 10 == 0:
				print('decrease learning rate!')
				learning_rate *= 0.5
				learning_rate = max(learning_rate, 0.0001)
			# loss = categorical_crossentropy
			loss = categorical_crossentropy
			#
			list_of_variables = self.model.trainable_variables
			print('loss: ' + str(float(loss(y, self.model(x)))))
			with tf.GradientTape() as tape:
				grads = tape.gradient(loss(y, self.model(x)), list_of_variables)
			num_grads_zero = 0
			num_grads_not_zero = 0
			num_grads_nan = 0
			num_grads_not_nan = 0
			pre_update = []
			new_weights = []
			for i in range(len(self.model.trainable_weights)):
				pre_update.append(np.array(self.model.trainable_weights[i][:]))
				num_grads_zero += np.sum(grads[i] == 0)
				num_grads_not_zero += np.sum(grads[i] != 0)
				num_grads_nan += np.sum(np.isnan(grads[i]))
				num_grads_not_nan += (np.size(grads[i]) - np.sum(np.isnan(grads[i])))
				# list_of_variables[i].assign_sub(0.001 * grads[i])
				new_weights.append(np.array(pre_update[i] - self.config['learning_rate'] * grads[i]))
			self.model.set_weights(new_weights)
			print('num_grads_zero: ' + str(num_grads_zero) + ', num_grads_not_zero:' + str(num_grads_not_zero))
			print('num_grads_nan: ' + str(num_grads_nan) + ', num_grads_not_nan:' + str(num_grads_not_nan))
			# code.interact(local=dict(globals(), **locals()))
			#
			post_update = self.model.trainable_weights
			is_equal = 0
			is_different = 0
			for i in range(len(pre_update)):
				is_equal += np.sum(pre_update[i] == post_update[i])
				is_different += np.sum(pre_update[i] != post_update[i])
			print('is_equal: ' + str(is_equal) + ', is_different:' + str(is_different))
			ratio_correct, ratio_total, ratio_invalid = accuracy(self.model, x, y, self.config)
			print('')
		ratio_correct, ratio_total, ratio_invalid = accuracy(self.model, x, y, self.config)
		print('ratio correct: ' + str(ratio_correct))
		#print('in train_on_one_example')
		#code.interact(local=dict(globals(), **locals()))
		return ratio_correct, ratio_total, ratio_invalid


	def unit_test(self, x_list, y_list):
		ratio_correct_list = []
		random_list = []
		for i in range(self.config['num_unit_tests']):
			n = len(x_list)
			s = int(random.uniform(0,n))
			random_list.append(s)
			x = x_list[s]
			y = y_list[s]
			ratio_correct_list.append(self.train_on_one_example(x, y))

	def train(self, train_data, val_data):
		meta_weights = []
		for i in range(len(self.model.trainable_weights)):
			meta_weights.append(np.array(self.model.trainable_weights[i][:]))
		for epoch in range(self.config['current_epoch'], self.config['max_epochs']):
			print('epoch: ' + str(epoch))
			meta_grads = []
			current_train_losses = []
			current_train_accuracies = []
			current_train_total_accuracies = []
			current_val_losses = []
			current_val_accuracies = []
			current_val_total_accuracies = []
			for sample_nr in range(len(train_data['x'])): #for sample_nr in range(self.config['meta_batch_size']):
				print(sample_nr)
				# code.interact(local=dict(globals(), **locals()))
				self.model.set_weights(meta_weights)
				loss = categorical_crossentropy
				x = train_data['x'][sample_nr]
				y = train_data['y'][sample_nr]
				x_val = train_data['x_val'][sample_nr]
				y_val = train_data['y_val'][sample_nr]
				meta_grads.append([])
				#
				list_of_variables = self.model.trainable_variables
				current_loss = float(loss(y, self.model(x)))
				with tf.GradientTape() as tape:
					grads = tape.gradient(loss(y, self.model(x)), list_of_variables)
				num_grads_zero = 0
				num_grads_not_zero = 0
				pre_update = []
				new_weights = []
				for i in range(len(self.model.trainable_weights)):
					pre_update.append(np.array(self.model.trainable_weights[i][:]))
					current_grad = np.array(grads[i])
					#meta_grads[-1].append(current_grad)
					num_grads_zero += np.sum(current_grad == 0)
					num_grads_not_zero += np.sum(current_grad != 0)
					new_weights.append(pre_update[i] - self.config['learning_rate'] * current_grad)
				del(grads)
				self.model.set_weights(new_weights)
				#
				post_update = self.model.trainable_weights
				is_equal = 0
				is_different = 0
				for i in range(len(pre_update)):
					is_equal += np.sum(pre_update[i] == post_update[i])
					is_different += np.sum(pre_update[i] != post_update[i])
				ratio_correct_val, ratio_total_val, ratio_invalid_val = accuracy(self.model, x_val, y_val, self.config)
				meta_loss = float(loss(y_val, self.model(x_val)))
				#
				current_train_losses.append(meta_loss)
				current_train_accuracies.append(ratio_correct_val)
				current_train_total_accuracies.append(ratio_total_val)
				#
				with tf.GradientTape() as tape:
					meta_grad = tape.gradient(loss(y_val, self.model(x_val)), list_of_variables)
				for i in range(len(meta_weights)):
					meta_grads[-1].append(np.array(meta_grad[i]))
				del(meta_grad)
				# apply meta step
				if len(meta_grads) >= self.config['meta_batch_size'] or sample_nr == len(train_data['x']) - 1:
					print('meta update')
					for i in range(len(meta_weights)):
						current_grad = meta_grads[0][i]
						for j in range(1,len(meta_grads)):
							current_grad += meta_grads[j][i]
						current_grad /= len(meta_grads)
						meta_weights[i] = meta_weights[i] - self.config['meta_learning_rate'] * current_grad
					meta_grads = []
				#
				if self.config['verbosity'] >= 1:
					ratio_correct, ratio_total, ratio_invalid = accuracy(self.model, x, y, self.config)
					print('loss: ' + str(current_loss))
					print('num_grads_zero: ' + str(num_grads_zero) + ', num_grads_not_zero:' + str(num_grads_not_zero))
					print('is_equal: ' + str(is_equal) + ', is_different:' + str(is_different))
					print('meta_loss: ' + str(meta_loss))
					print('ratio_correct_meta: ' + str(ratio_correct_val))
					print('ratio_total_meta: ' + str(ratio_total_val))
					print('')
					print('')
			#
			for sample_nr in range(len(val_data['x'])): #for sample_nr in range(self.config['meta_batch_size']):
				print(sample_nr)
				# code.interact(local=dict(globals(), **locals()))
				self.model.set_weights(meta_weights)
				loss = categorical_crossentropy
				x = train_data['x'][sample_nr]
				y = train_data['y'][sample_nr]
				x_val = train_data['x_val'][sample_nr]
				y_val = train_data['y_val'][sample_nr]
				meta_grads.append([])
				#
				list_of_variables = self.model.trainable_variables
				print('loss: ' + str(float(loss(y, self.model(x)))))
				with tf.GradientTape() as tape:
					grads = tape.gradient(loss(y, self.model(x)), list_of_variables)
				num_grads_zero = 0
				num_grads_not_zero = 0
				pre_update = []
				new_weights = []
				for i in range(len(self.model.trainable_weights)):
					pre_update.append(np.array(self.model.trainable_weights[i][:]))
					current_grad = np.array(grads[i])
					#meta_grads[-1].append(current_grad)
					num_grads_zero += np.sum(current_grad == 0)
					num_grads_not_zero += np.sum(current_grad != 0)
					new_weights.append(pre_update[i] - self.config['learning_rate'] * current_grad)
				del(grads)
				self.model.set_weights(new_weights)
				#
				post_update = self.model.trainable_weights
				is_equal = 0
				is_different = 0
				for i in range(len(pre_update)):
					is_equal += np.sum(pre_update[i] == post_update[i])
					is_different += np.sum(pre_update[i] != post_update[i])
				ratio_correct_val, ratio_total_val, ratio_invalid_val = accuracy(self.model, x_val, y_val, self.config)
				#
				current_train_losses.append(meta_loss)
				current_train_accuracies.append(ratio_correct_val)
				current_train_total_accuracies.append(ratio_total_val)
				#
				if self.config['verbosity'] >= 1:
					ratio_correct, ratio_total, ratio_invalid = accuracy(self.model, x, y, self.config)
					print('num_grads_zero: ' + str(num_grads_zero) + ', num_grads_not_zero:' + str(num_grads_not_zero))
					print('is_equal: ' + str(is_equal) + ', is_different:' + str(is_different))
					print('meta_loss: ' + str(float(loss(y_val, self.model(x_val)))))
					print('ratio_correct_val: ' + str(ratio_correct_val))
					print('ratio_total_val: ' + str(ratio_total_val))
					print('')
					print('')
		#
		self.config['current_epoch'] += 1
		open('saved_models/' + self.config['name'] + '/config.json', 'w').write(json.dumps(self.config, sort_keys=True, indent=4))
		self.model.save_weights('saved_models/' + self.config['name'] + '/weights/' + str(self.config['current_epoch']))