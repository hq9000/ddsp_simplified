from typing import Dict

import numpy as np


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

        pass