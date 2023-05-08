'''A script that loads some classified skeleton data and trains weights using it.'''
import json
from os.path import join
from config import KINECT_EXPERIMENT_DIR
from anneal import (generate_distances,
                    KinectWeightsProblem,
                    score_weights,
                    split_items,
                    load_option_info,
                    load_skeleton_data)
from normal import (normalize_scale, normalize_origin)
from random import shuffle
from random import seed
from time import clock

#if __name__ == '__main__': # Emacs doesn't like this.

OPTION_INFO = load_option_info('option.json')

## Build path to file.
EXPERIMENT_FILE_PATH  = join(
    KINECT_EXPERIMENT_DIR,
    OPTION_INFO['dir_name'],
    OPTION_INFO['file_name']
)

## Open file and extract JSON frames data.
FRAMES = normalize_scale(normalize_origin(load_skeleton_data(EXPERIMENT_FILE_PATH), 'HipCenter'),
                         'HipCenter', 'Head')

## Get the joint distances from the frame data.
DISTANCES = map(generate_distances, FRAMES)

## Split the distance data. Use one half for
## training and the other half for testing.
seed(OPTION_INFO['shuffle_seed'])
shuffle(DISTANCES)
TRAINING_DISTANCES, TESTING_DISTANCES = split_items(DISTANCES, 0.5)

seed(OPTION_INFO['training_seed'])
## Train the weights.
WEIGHTS, _ = KinectWeightsProblem(TRAINING_DISTANCES).anneal()

## Test the weights and display info on how well they work.
SCORES = score_weights(TESTING_DISTANCES, WEIGHTS)
print 'Good:\t%d' % SCORES['good']
print 'Bad:\t%d' % SCORES['bad']
print 'Error:\t%d' % SCORES['error']
print 'Score:\t%.4f%%' % (SCORES['accuracy'] * 100)
print json.dumps(WEIGHTS)
