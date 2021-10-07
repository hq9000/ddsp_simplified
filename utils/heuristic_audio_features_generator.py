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

        if MIDI_FEATURE_PITCH not in midi_features:
            raise Exception('a midi features dict is missing pitch data')

        if MIDI_FEATURE_VELOCITY not in midi_features:
            raise Exception('a midi features dict is missing velocity data')

        return {
            INPUT_FEATURE_F0_HZ: self._generate_f0_hz_by_pitches(midi_features[MIDI_FEATURE_PITCH]),
            INPUT_FEATURE_LOUDNESS_DB: self._generate_loudness_db_by_velocity(midi_features[MIDI_FEATURE_VELOCITY])
        }

    def _generate_f0_hz_by_pitches(self, pitch_array: np.ndarray) -> np.ndarray:
        return (2 ** ((pitch_array - 69) / 12)) * 440

    def _generate_loudness_db_by_velocity(self, velocity_array: np.ndarray) -> np.ndarray:
        return velocity_array / 127

