#!/usr/bin/python
'''A script that loads data and trains different models using them.'''
import math
from os.path import join
from kinect_learning import (joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)

## Build path to file.
DATA_DIR = 'data'
FILE_NAME = 'up-down.csv'
FILE_PATH = join(DATA_DIR, FILE_NAME)

left_right_col = joints_collection('left-right')
sit_stand_col = joints_collection('sit-stand')
turning_col = joints_collection('turning')
bending_col = joints_collection('bending')
up_down_col = joints_collection('up-down')
all_col = joints_collection('all')

COLLECTION = up_down_col
print("Printing scores of small collection...")
print("Collection includes", COLLECTION)
noise = False
X, y = load_data(FILE_PATH, COLLECTION, noise)['positions'], load_data(FILE_PATH, COLLECTION, noise)['labels']

test_size = [0.4, 0.6, 0.8]
col_size = len(COLLECTION)
n_neighbors = (int)(math.sqrt(col_size))
kernel = ['linear', 'rbf', 'poly']
n_epochs = 300

for size in test_size:
	n_estimators = int(len(X)*(1 - size))

	'''
		for ker in kernel:
		if size != 0.4:
			break
		else:
			print("SVM with test_size =", size, "and", ker, "kernel:", SVM(X, y, size, ker))
	'''
	print("SVM with test_size =", size, ":", SVM(X, y, size, 'linear'))
	print("Gaussian NB with test_size =", size, ":", Gaussian_NB(X, y, size))
	print("Random Forest with test_size =", size, ":", Random_Forest(X, y, size, n_estimators))
	print("AdaBoost with test_size =", size, ":", AdaBoost(X, y, size, n_estimators))
	print("Neural Network with test_size =", size, ":", Neural_Network(X, y, size, col_size, n_epochs))
	print("Nearest Neighbor with test_size =", size, ":", Knn(X, y, size, n_neighbors))
	print("")

print("Printing scores of small collection with noise data...")
noise = True
X, y = load_data(FILE_PATH, COLLECTION, noise)['positions'], load_data(FILE_PATH, COLLECTION, noise)['labels']
test_size = 0.4
n_estimators = int(len(X)*(1 - test_size))
print("SVM with noise data:", SVM(X, y, test_size, 'linear'))
print("Random Forest with noise data:", Random_Forest(X, y, test_size, n_estimators))
print("AdaBoost with noise data:", AdaBoost(X, y, test_size, n_estimators))
print("Neural Network with noise data:", Neural_Network(X, y, test_size, col_size, n_epochs))
print("Gaussian NB with noise data:", Gaussian_NB(X, y, test_size))
print("Nearest Neighbor with noise data", Knn(X, y, size, n_neighbors))

COLLECTION = all_col
col_size = len(	COLLECTION)
n_neighbors = (int)(math.sqrt(col_size))
print("")
print("Printing scores of all collection...")
noise = False
X, y = load_data(FILE_PATH, COLLECTION, noise)['positions'], load_data(FILE_PATH, COLLECTION, noise)['labels']
print("SVM with all collection:", SVM(X, y, test_size, 'linear'))
print("Random Forest with all collection:", Random_Forest(X, y, test_size, n_estimators))
print("AdaBoost with all collection:", AdaBoost(X, y, test_size, n_estimators))
print("Neural Network with all collection:", Neural_Network(X, y, test_size, col_size, n_epochs))
print("Gaussian NB with all collection:", Gaussian_NB(X, y, test_size))
print("Nearest Neighbor with all data", Knn(X, y, size, n_neighbors))
