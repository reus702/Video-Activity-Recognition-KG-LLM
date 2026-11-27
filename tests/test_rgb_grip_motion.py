import unittest

import cv2
import numpy as np

from src.rgb_grip_motion import wrist_motion_cue


class WristMotionTests(unittest.TestCase):
    def test_stationary_center_against_moving_scene(self):
        rng = np.random.default_rng(2)
        texture = rng.integers(0, 256, (180, 320), dtype=np.uint8)
        before = cv2.GaussianBlur(texture, (5, 5), 0)
        shift = np.float32([[1, 0, 7], [0, 1, 4]])
        after = cv2.warpAffine(before, shift, (320, 180))

        self.assertIsNone(wrist_motion_cue(before, before)["cue"])
        self.assertIsNone(wrist_motion_cue(before, after)["cue"])

        after[74:138, 134:182] = before[74:138, 134:182]
        self.assertEqual(wrist_motion_cue(before, after)["cue"], "possible_held_piece")
