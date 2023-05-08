import unittest
import json
from os.path import join
from normal import normalize_origin, normalize_scale
from config import KINECT_EXPERIMENT_DIR
from anneal import distance, json_hash_to_vector, load_skeleton_data

class TestNormal(unittest.TestCase):
    def setUp(self):
        option_info = { "dir_name" : "11_11_2015_2_51_30_PMHalfProfile",
          "file_name" : "leftLabels.csv",
          "training_ratio" : "0.5"
        }

        ## Build path to file.
        experiment_file_path  = join(
            KINECT_EXPERIMENT_DIR,
            option_info['dir_name'],
            option_info['file_name']
        )

        ## Open file and extract JSON frames data.
        self.frames = load_skeleton_data(experiment_file_path)
    def test_normalize_origin(self):
        'Check that the origin joint is zero.'
        center_joint = 'HipCenter'
        for frame in normalize_origin(self.frames, center_joint):
            self.assertEqual(
                frame['jointPositions']['jointPositionDict'][center_joint],
                (0.0, 0.0, 0.0))
               
    def test_normalize_scale(self):
        ratios = []
        check_frame = self.frames[0]['jointPositions']['jointPositionDict']
        other_check_frame = normalize_scale(self.frames[:1], 'HipCenter', 'Head')[0]['jointPositions']['jointPositionDict']
        for first_joint in check_frame.keys():
            for second_joint in other_check_frame.keys():
                if first_joint == second_joint:
                    continue
                first_check_vector = check_frame[first_joint]
                second_check_vector = check_frame[second_joint]
                first_other_check_vector = other_check_frame[first_joint]
                second_other_check_vector = other_check_frame[second_joint]
                check_ratio = distance(
                    first_check_vector,
                    second_check_vector)
                other_check_ratio = distance(
                    first_other_check_vector,
                    second_other_check_vector)
                ratios.append(check_ratio / other_check_ratio)
        head = ratios[0]
        for i, ratio in enumerate(ratios):
            self.assertTrue(abs(ratio - head) < 0.0000001, (ratio, head))
            
            
        
