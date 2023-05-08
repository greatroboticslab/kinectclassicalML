'''A list of global constants used throughout the program.
They should never be edited during runtime.'''

from os import environ
from os.path import join
import json

APP_DATA_DIR_NAME = environ['APPDATA']
KINECT_EXPERIMENT_DIR_NAME = 'KinectExperiment'
KINECT_EXPERIMENT_DIR = join(APP_DATA_DIR_NAME, 'KinectExperiment')

RELEVANT_JOINTS_FILE_NAME = 'RelevantJoints.json'
with file(join(KINECT_EXPERIMENT_DIR, RELEVANT_JOINTS_FILE_NAME)) as f_obj:
    RELEVANT_JOINTS = json.load(f_obj)

RELEVANT_JOINT_PAIRS = [(joint_one, joint_two)
    for joint_one in RELEVANT_JOINTS
    for joint_two in RELEVANT_JOINTS
    if not (joint_one >= joint_two)]
