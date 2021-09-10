from typing import Dict

import numpy as np

from feature_names import INPUT_FEATURE_F0_HZ, INPUT_FEATURE_LOUDNESS_DB


class HeuristicAudioFeaturesGenerator:
    def generate(self, midi_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Generate a dict of heuristically generated audio features based on midi features

        Generates the following audio features:

        - "f0_hz"
        - "loudness_db"

        based on the following midi features:

        - "pitch"
        - "velocity"

        the length of output arrays is the same as the one of input arrays.
        """
        length = len(list(midi_features.items())[0][1])

        return {
            INPUT_FEATURE_F0_HZ: np.zeros(dtype=np.float32, shape=(length,)),
            INPUT_FEATURE_LOUDNESS_DB: np.zeros(dtype=np.float32, shape=(length,))
        }
