import unittest

import numpy as np
from numpy import array

from feature_names import MIDI_FEATURE_PITCH, MIDI_FEATURE_VELOCITY, FEATURE_F0_HZ, INPUT_FEATURE_F0_HZ, \
    INPUT_FEATURE_LOUDNESS_DB
from utils.heuristic_audio_features_generator import HeuristicAudioFeaturesGenerator


class MyTestCase(unittest.TestCase):
    def test_something(self):
        generator = HeuristicAudioFeaturesGenerator()
        size = 5
        res = generator.generate({
            MIDI_FEATURE_PITCH: np.full(size, 65),
            MIDI_FEATURE_VELOCITY: np.full(size, 67)
        })

        expected_res = {'f0_hz': array([349.22823143, 349.22823143, 349.22823143, 349.22823143,
                                        349.22823143]),
                        'loudness_db': array([0.52755906, 0.52755906, 0.52755906, 0.52755906, 0.52755906])}

        self.assertTrue(np.isclose(res[INPUT_FEATURE_F0_HZ], expected_res[INPUT_FEATURE_F0_HZ]).any())
        self.assertTrue(np.isclose(res[INPUT_FEATURE_LOUDNESS_DB], expected_res[INPUT_FEATURE_LOUDNESS_DB]).any())

if __name__ == '__main__':
    unittest.main()
