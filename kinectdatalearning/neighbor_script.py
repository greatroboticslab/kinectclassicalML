'''A script that loads some classified skeleton data and trains a nearest neighbor classifier using it.'''

import json
from os.path import join
from config import KINECT_EXPERIMENT_DIR
import random
from anneal import (generate_distances,
                    KinectWeightsProblem,
                    score_weights,
                    split_items,
                    load_option_info,
                    load_skeleton_data,
                    tweak)
from simanneal import Annealer
from normal import normalize_scale, normalize_origin
from copy import deepcopy

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

class KinectWeightsProblem(Annealer):
    'A class with methods defined for generating weights for classifying joint data.'
    def __init__(self, frames):
        self.frames = frames
        self.state = {joint : random.uniform(0.0, 0.5)
                      for joint in self.frames[0]['jointPositions']['jointPositionDict'].keys()}
        self.ideal_model = {joint : (0, 0, 0) for joint in self.frames[0]['jointPositions']['jointPositionDict'].keys()}
        for frame in self.frames:
            skeleton = frame['jointPositions']['jointPositionDict']
            if frame['label'] == 1:
                for joint in skeleton.keys():
                    self.ideal_model[joint] = (
                        self.ideal_model[joint][0] + skeleton[joint][0],
                        self.ideal_model[joint][1] + skeleton[joint][1],
                        self.ideal_model[joint][2] + skeleton[joint][2],
                    )
        average_count = len([None for frame in self.frames if frame['label'] == 1])
        for joint in self.ideal_model.keys():
            self.ideal_model[joint] = tuple(item / average_count for item in self.ideal_model[joint])
    def move(self):
        self.state = {joint : abs(tweak(self.state[joint])) for joint in self.state.keys()}
    def fitness(self, skeleton):
        sum = 0
        for joint in skeleton.keys():
            v1 = skeleton[joint]
            v2 = self.ideal_model[joint]
            sum += self.state[joint] * ((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2)
        return sum
    def energy(self):
        yes_count = 0
        yes_sum = 0
        no_count = 0
        no_sum = 0
        for frame in self.frames:
            skeleton = frame['jointPositions']['jointPositionDict']
            score = self.fitness(skeleton)
            if frame['label'] == 1:
                yes_sum += score
                yes_count += 1
            elif frame['label'] == -1:
                no_sum += score
                no_count += 1
            elif frame['label'] == 0:
                pass
            else:
                raise 'bad label %s', frame['label']
        yes_score_average = float(yes_sum) / yes_count
        no_score_average = float(no_sum) / no_count
        pivot_score = (yes_score_average + no_score_average) / 2

        correct_count = 0
        incorrect_count = 0
        for frame in self.frames:
            score = self.fitness(skeleton)
            if frame['label'] == 1 and score < pivot_score:
                correct_count += 1
            elif frame['label'] == 0 and score > pivot_score:
                correct_count += 1
            elif frame['label'] == 1 and score > pivot_score:
                incorrect_count += 1
            elif frame['label'] == 0 and score < pivot_score:
                incorrect_count += 1
        total_count = correct_count + incorrect_count + 1 #TODO remove + 1
        return float(correct_count) / total_count

## Split the distance data. Use one half for
## training and the other half for testing.
random.seed(OPTION_INFO['shuffle_seed'])
random.shuffle(FRAMES)
TRAINING_FRAMES, TESTING_FRAMES = split_items(FRAMES, 0.5)

random.seed(OPTION_INFO['training_seed'])
## Train the weights.
WEIGHTS, _ = KinectWeightsProblem(TRAINING_FRAMES).anneal()

## Test the weights and display info on how well they work.
SCORES = score_weights(TESTING_DISTANCES, WEIGHTS)
print 'Good:\t%d' % SCORES['good']
print 'Bad:\t%d' % SCORES['bad']
print 'Error:\t%d' % SCORES['error']
print 'Score:\t%.4f%%' % (SCORES['accuracy'] * 100)
print json.dumps(WEIGHTS)
