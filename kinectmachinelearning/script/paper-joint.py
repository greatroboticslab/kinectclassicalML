#!/usr/bin/python
'''A script that loads data and trains different models using them.'''
import math
from os.path import join
from kinect_learning import (joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)

## Build path to file.
DATA_DIR = 'data'
FILE_NAME = 'left-right.csv'
FILE_PATH = join(DATA_DIR, FILE_NAME)

left_right_col = joints_collection('left-right')
sit_stand_col = joints_collection('sit-stand')
turning_col = joints_collection('turning')
bending_col = joints_collection('bending')
up_down_col = joints_collection('up-down')
all_col = joints_collection('all')

COLLECTION = left_right_col
print("Printing scores of small collection...")
print("Collection includes", COLLECTION)
print("Printing scores of small collection with noise data...")
noise = True
X, y = load_data(FILE_PATH, COLLECTION, noise)['positions'], load_data(FILE_PATH, COLLECTION, noise)['labels']
test_size = 0.4
col_size = len(COLLECTION)
n_neighbors = (int)(math.sqrt(col_size))
kernel = ['linear', 'rbf', 'poly']
n_epochs = 300
n_estimators = int(len(X)*(1 - test_size))
print("SVM with noise data:", SVM(X, y, test_size, 'linear'))
#print("Random Forest with noise data:", Random_Forest(X, y, test_size, n_estimators))
print("AdaBoost with noise data:", AdaBoost(X, y, test_size, n_estimators))
print("Neural Network with noise data:", Neural_Network(X, y, test_size, col_size, n_epochs))
print("Gaussian NB with noise data:", Gaussian_NB(X, y, test_size))


