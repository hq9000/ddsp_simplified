from typing import List, Optional

import numpy as np

from ddsp_simplified.feature_names import MIDI_FEATURE_VELOCITY, MIDI_FEATURE_PITCH
from ddsp_simplified.models import Autoencoder
from ddsp_simplified.utils.heuristic_audio_features_generator import HeuristicAudioFeaturesGenerator
from ddsp_simplified.utils.midi_loader import MidiLoader


def synthesize_audio_by_midi(
        model: Autoencoder,
        path_to_midi_file: str,
        frame_rate: int,
        length_of_audio_seconds: float,
        midi_feature_names: List[str]
) -> Optional[np.ndarray]:
    midi_loader = MidiLoader()
    midi_feature_names = get_midi_feature_names_augmented_with_pitch_and_velocity(midi_feature_names)

    midi_features = midi_loader.load(
        midi_file_name=path_to_midi_file,
        frame_rate=frame_rate,
        audio_length_seconds=length_of_audio_seconds,
        only_these_features=midi_feature_names,
        raise_on_failure=False)

    if midi_features is None:
        return None

    audio_features_generator = HeuristicAudioFeaturesGenerator()
    heuristic_audio_features = audio_features_generator.generate(midi_features)

    # we do not need pitch and velocity as midi feature as they
    # have already been encoded as heuristic audio features
    del midi_features[MIDI_FEATURE_VELOCITY]
    del midi_features[MIDI_FEATURE_PITCH]

    features = {
        **heuristic_audio_features,
        **midi_features
    }

    return model.transfer_timbre(features)


def get_midi_feature_names_augmented_with_pitch_and_velocity(midi_feature_names: List[str]) -> List[str]:
    midi_feature_names = list(midi_feature_names)

    for unexpected_midi_feature in [MIDI_FEATURE_PITCH, MIDI_FEATURE_VELOCITY]:
        if unexpected_midi_feature in midi_feature_names:
            raise Exception(
                unexpected_midi_feature + " is in the original list of midi features. This is not expected!")

    midi_feature_names.append(MIDI_FEATURE_PITCH)
    midi_feature_names.append(MIDI_FEATURE_VELOCITY)

    return midi_feature_names
