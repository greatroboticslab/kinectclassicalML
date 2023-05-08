'''Contains utility functions for loading and training skeleton data.'''

from itertools import izip
import math
import json
from time import clock
from simanneal import Annealer
from random import uniform
from config import RELEVANT_JOINT_PAIRS

from sys import argv

def json_hash_to_vector(vector_str):
    'Convert a JSON hash representing a vector to a collection of numbers.'
    return map(float, (vector_str['X'], vector_str['Y'], vector_str['Z']))

def label_switch(enum_value):
    'Convert a JSON integer value so that; Yes => 1, No => -1, Null => 0'
    return {0 : 1, 1 : -1, 2 : 0}[enum_value]

def load_skeleton_data(file_name):
    with open(file_name) as f_obj:
        skeleton_data = json.load(f_obj)
    for frame in skeleton_data:
        for joint_type in frame['jointPositions']['jointPositionDict'].keys():
            vector = json_hash_to_vector(frame['jointPositions']['jointPositionDict'][joint_type])
            frame['jointPositions']['jointPositionDict'][joint_type] = vector
        frame['label'] = label_switch(frame['label'])
    return skeleton_data

def distance(initial, terminal):
    'Return the distance between two positions.'
    return math.sqrt(sum((t - i) ** 2 for i, t in izip(initial, terminal)))

def generate_distances(frame):
    'Generates distances between relevant joints.'
    positions = frame[u'jointPositions'][u'jointPositionDict']
    distance_frame = {}
    for initial_joint_name, terminal_joint_name in RELEVANT_JOINT_PAIRS:
        initial_joint_position = positions[initial_joint_name]
        terminal_joint_position = positions[terminal_joint_name]
        joint_key = (initial_joint_name, terminal_joint_name)
        distance_frame[joint_key] = distance(
            initial_joint_position,
            terminal_joint_position
        )
    distance_frame['label'] = frame['label']
    return distance_frame

def difference(joint_datum, weights):
    'Return the difference between the approximate solution with weights and the'
    accum = 0 
    for joint_name, weight in izip(RELEVANT_JOINT_PAIRS, weights):
        accum += (joint_datum[joint_name] * weight)
    return accum + joint_datum['label']

def tweak(value):
    'Tweak the given value.'
    return value + uniform(-0.05, 0.05)
    
class KinectWeightsProblem(Annealer):
    'A class with methods defined for generating weights for classifying joint data.'
    def __init__(self, joint_data):
        self.joint_data = joint_data
        self.state = [uniform(-0.5, 0.5) for _ in RELEVANT_JOINT_PAIRS]
    def move(self):
        self.state = [tweak(weight) for weight in self.state]
    def energy(self):
        fitness = sum(difference(joint_datum, self.state) ** 2 for joint_datum in self.joint_data)
        regularization_rate = 2
        regularization = regularization_rate * sum(state ** 2 for state in self.state)
        return fitness + regularization

def signum(n):
    'Return a number representing the sign of the given number.'
    if n == 0:
        return 0
    elif n < 0:
        return -1
    elif n > 0:
        return 1

def split_items(items, ratio):
    '''Split a collection into two collections where each contains randomly selected data.
The ratio determines how many items each list holds.
A list with 10 items and a ratio of 0.5 would return two lists with 5 items each.
A list with 10 items and a ratio of 0.3 would return two lists with 3 items in the first list  and 7 items in the second list.'''
    pivot = int(ratio * len(items))
    return items[:pivot], items[pivot:]

def score_weights(distances, weights):
    'Score the weights and return score data.'
    good_score = 0
    bad_score = 0
    error_score = 0
    for frame_datum in distances:
        sign = signum(difference(frame_datum, weights))
        label = frame_datum['label']
        if sign == -1 and label == 1:
            bad_score += 1
        elif sign == 1 and label == -1:
            bad_score += 1
        elif sign == label:
            good_score += 1
        else:
            error_score += 1
    return {
        'good' : good_score,
        'bad' : bad_score,
        'error' : error_score,
        'accuracy' : (1.0 * good_score / (good_score + bad_score))
    }

class BadOptionInfoException(Exception):
    pass

class BadOptionKeysException(BadOptionInfoException):
    'Raised when a bad option file with missing keys is loaded.'
    def __init__(self, keys):
        self.keys = keys
    def __repr__(self):
        print 'Missing the keys: %s' % (self.keys,)

class BadOptionValuesException(BadOptionInfoException):
    'Raised when a bad option file with invalid values is loaded.'
    def __init__(self, key_value_pairs):
        self.key_value_pairs = key_value_pairs
    def __repr__(self):
        print 'Invalid values: %s' % (self.key_value_pairs,)

def load_option_info(filename):
    'Load option info from the given file.'
    with file(filename) as f_obj:
        option_info = json.load(f_obj)
        missing_required_keys = []
        bad_values = []
        if 'dir_name' not in option_info:
            missing_required_value += 'dir_name'
        if 'file_name' not in option_info:
            missing_required_keys += 'file_name'
        if 'training_ratio' not in option_info:
            missing_required_keys += 'training_ratio'
        if 'shuffle_seed' not in option_info:
            option_info['shuffle_seed'] = clock()
        else:
            try:
                option_info['shuffle_seed'] = float(option_info['shuffle_seed'])
            except ValueError:
                bad_values.append(('shuffle_seed', option_info['shuffle_seed']))
        if 'training_seed' not in option_info:
            option_info['training_seed'] = clock()
        else:
            try:
                option_info['training_seed'] = float(option_info['training_seed'])
            except ValueError:
                bad_values.append(('training_seed', option_info['training_seed']))
        if missing_required_keys:
            raise BadOptionKeysException(missing_required_keys)
        if bad_values:
            raise BadOptionValuesException(bad_values)
    return option_info

def auto_load_option_info():
    'Load option info from the filename found in the first shell argument.'
    usage = 'usage: %s config-file'
    try:
        return load_option_info(argv[1])
    except BadOptionInfoException as e:
        print e
        print usage
        exit()
    except IndexError:
        print usage
        exit()
