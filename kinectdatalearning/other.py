from anneal import load_option_info
from os.path import join
from config import KINECT_EXPERIMENT_DIR
import json
from sklearn.neighbors.nearest_centroid import NearestCentroid
import random

import numpy as np

def magnitude(vector):
    return np.sqrt(vector.dot(vector))

class SkeletonFrame(object):
    'A frame of classified skeleton data.'
    def __init__(self, joint_dict, label):
        self.positions = joint_dict
        self.label = label
        self.distances = {}
        self._generate_distances(self.positions)
    def _generate_distances(self, joint_dict):
        'Return a dictionary that maps joint pairs to distances.'
        joint_pairs = [(joint1, joint2)
         for joint1 in joint_dict.keys()
         for joint2 in joint_dict.keys()]
        for (joint1, joint2) in joint_pairs:
            difference = joint_dict[joint1] - joint_dict[joint2]
            self.distances[(joint1, joint2)] = magnitude(difference)
    def normalize_origin(self, origin_joint):
        'Normalize the position of the skeleton so that the provided joint is on the origin.'
        for joint in self.positions.keys():
            self.positions[joint] -= self.positions[origin_joint]
    def normalize_scale(self, joint_pair):
        'Normalize the scale of the skeleton so the provided joint pair has a unit magnitude and every other joint pair is scaled.'
        joint1, joint2 = joint_pair
        difference = self.positions[joint1] - self.positions[joint2]
        scale_factor = 1 / magnitude(difference)
        for joint in self.positions.keys():
            self.positions[joint] *= scale_factor
        self._generate_distances(self.positions)


def load_skeleton_data(file_name):
    with open(file_name) as f_obj:
        skeleton_data = json.load(f_obj)
    frames = []
    for frame in skeleton_data:
        joints = frame['jointPositions']['jointPositionDict']
        frames.append(SkeletonFrame(
            { joint : np.array(map(float, [joints[joint]['X'],
                                           joints[joint]['Y'],
                                           joints[joint]['Z']]))
                          for joint in joints.keys() },
            frame['label']))
    return frames


#load
OPTION_INFO = load_option_info('option.json')

## Build path to file.
EXPERIMENT_FILE_PATH  = join(
    KINECT_EXPERIMENT_DIR,
    OPTION_INFO['dir_name'],
    OPTION_INFO['file_name']
)

times_better, times_wrong, times_same = 0, 0, 0
for i in xrange(1000):
    last_accuracy = 0
    for scale in [False, True]:
        ## Open file and extract JSON frames data.
        FRAMES = load_skeleton_data(EXPERIMENT_FILE_PATH)
        for frame in FRAMES:
            if scale:
                frame.normalize_origin('HipCenter')
            frame.normalize_scale(('HipCenter', 'Head'))
            pass
        #train

        np.random.seed(i)
        random.seed(i)

        random.shuffle(FRAMES)
        data = np.array([[frame.distances[joint] for joint in frame.distances.keys()]
                for frame in FRAMES])
        target = np.array([frame.label for frame in FRAMES])
        indices = np.random.permutation(len(data))
        data_train = data[indices[:-len(data)/2]]
        target_train = target[indices[:-len(data)/2]]
        data_test = data[indices[-len(data)/2:]]
        target_test = target[indices[-len(data)/2:]]

        knn = NearestCentroid()
        knn.fit(data_train, target_train)
        accuracy = sum(1
                       for (actual, correct) in zip(knn.predict(data_test), target_test)
                       if actual == correct) / float(len(target_test))
        if scale:
            if accuracy > last_accuracy: times_better += 1
            elif accuracy < last_accuracy: times_wrong += 1
            else: times_same += 1
        last_accuracy = accuracy

print times_better / float(times_better + times_wrong + times_same)
print (times_better, times_wrong, times_same)
        

