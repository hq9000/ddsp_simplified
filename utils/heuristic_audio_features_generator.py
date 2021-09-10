from typing import Dict

import numpy as np

from feature_names import INPUT_FEATURE_F0_HZ, INPUT_FEATURE_LOUDNESS_DB, MIDI_FEATURE_PITCH, MIDI_FEATURE_VELOCITY


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
            INPUT_FEATURE_F0_HZ: self._generate_f0_hz_by_pitches(midi_features[MIDI_FEATURE_PITCH]),
            INPUT_FEATURE_LOUDNESS_DB: self._generate_loudness_db_by_velocity(midi_features[MIDI_FEATURE_VELOCITY])
        }

    def _generate_f0_hz_by_pitches(self, pitch_array: np.ndarray) -> np.ndarray:
        # todo
        pass

    def _generate_loudness_db_by_velocity(self, velocity_array: np.ndarray) -> np.ndarray:
        # todo
        pass

