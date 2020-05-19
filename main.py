import argparse
import code  # code.interact(local=dict(globals(), **locals()))
import os
import json

# imports from own libraries
from dataloader import load_data
from models import ARCModel

parser = argparse.ArgumentParser()

parser.add_argument('--load_from_path', default=None, help='How much logs shall be printed?')
parser.add_argument('--verbosity', type=int, default=2, help='How much logs shall be printed?')
parser.add_argument('--meta_batch_size', type=int, default=5, help='How much logs shall be printed?')
parser.add_argument('--meta_learning_rate', type=float, default=0.0001, help='How big is the initial meta learning rate?')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='How big is the initial learning rate?')
parser.add_argument('--num_unit_tests', type=int, default=1, help='How many data points shall be unit tested?')
parser.add_argument('--model_type', default='transformer', help='which type of model? options: ["fc","conv","transformer"]')

args = parser.parse_args()

if args.load_from_path != None:
	config_string = open(args.load_from_path, 'r').read()
	global_config = json.loads(config_string)
	global_config['is_loaded'] = True
else:
	global_config = {
		'verbosity' : args.verbosity,
		'meta_batch_size' : args.meta_batch_size,
		'meta_learning_rate' : args.meta_learning_rate,
		'learning_rate' : args.learning_rate,
		'num_unit_tests' : args.num_unit_tests,
		'model_type' : args.model_type,
		'model_name' : None,
		'is_loaded' : False,
		'current_epoch' : 0,
		'max_epochs' : 100,
		'train_accuracies' : [0.0],
		'train_total_accuracies' : [0.0],
		'train_losses' : [float('inf')],
		'val_accuracies' : [0.0],
		'val_total_accuracies' : [0.0],
		'val_losses' : [float('inf')]
	}

def main():
	train_data = load_data()
	val_data = load_data(path='data/evaluation')
	model = ARCModel(global_config)
	#model.unit_test(train_data['x'], train_data['y'])
	model.train(train_data, val_data)
	code.interact(local=dict(globals(), **locals()))

main()