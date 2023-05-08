
from copy import deepcopy
from math import sqrt
from config import KINECT_EXPERIMENT_DIR
from anneal import json_hash_to_vector

def normalize_origin(frames, center_joint):
    'Return a new set of frames where every frame is translated so the origin is at the given center joint.'
    frames = deepcopy(frames)
    for i, frame in enumerate(frames):
        center = frame['jointPositions']['jointPositionDict'][center_joint]
        for jointType in frame['jointPositions']['jointPositionDict'].keys():
            vector = frame['jointPositions']['jointPositionDict'][jointType]
            new_vector = (
                vector[0] - center[0],
                vector[1] - center[1],
                vector[2] - center[2],
            )
            frames[i]['jointPositions']['jointPositionDict'][jointType] = new_vector
    return frames

def normalize_scale(frames, joint_one, joint_two):
    'Return a new set of frames where every frame is scaled so that the distance between the given joints is 1 and the ratios of the distances between joints is unchanged.'
    frames = deepcopy(frames)
    for frame in frames:
        for i, frame in enumerate(frames):
            first_vector = frame['jointPositions']['jointPositionDict'][joint_one]
            second_vector = frame['jointPositions']['jointPositionDict'][joint_two]
            unit_length = sqrt(
                (first_vector[0] - second_vector[0])**2
                + (first_vector[1] - second_vector[1])**2
                + (first_vector[2] - second_vector[2])**2)
            for jointType in frame['jointPositions']['jointPositionDict'].keys():
                vector = frame['jointPositions']['jointPositionDict'][jointType]
                new_vector = (
                    vector[0] / unit_length,
                    vector[1] / unit_length,
                    vector[2] / unit_length,
                )
                frames[i]['jointPositions']['jointPositionDict'][jointType] = new_vector
    return frames


