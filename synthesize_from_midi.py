import argparse
from typing import List

import librosa
import numpy as np
import yaml

from models import SupervisedAutoencoder, Autoencoder
from train_utils import make_supervised_model
from utils.heuristic_audio_features_generator import HeuristicAudioFeaturesGenerator
from utils.midi_loader import MidiLoader
import scipy.io.wavfile as wavfile


def main():
    parser = argparse.ArgumentParser(description='Script to synthesize audio using trained model and midi file.')
    parser.add_argument('-c', '--config-path', type=str, required=True, help='Path to config file.')
    parser.add_argument('-m', '--midi-file', type=str, default='', help='Path to the input midi file.')
    parser.add_argument('-o', '--output-dir', type=str, default='', help='Output audio directory.')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))

    frame_rate: int = config['audio']['frame_rate']
    model = _create_model_and_load_weights(config)
    midi_feature_names: List[str] = config['data']['midi_features']

    audio = _synthesize_audio_by_midi(model,
                                      path_to_midi_file=args.midi_file,
                                      frame_rate=frame_rate,
                                      length_of_audio_seconds=10,
                                      midi_feature_names=midi_feature_names)

    _write_audio_to_file(audio, '/tmp/hello_world.wav')


def _create_model_and_load_weights(config) -> SupervisedAutoencoder:
    model = make_supervised_model(config)
    model.load_weights(config['model']['path'])
    return model


def _synthesize_audio_by_midi(
        model: Autoencoder,
        path_to_midi_file: str,
        frame_rate: int,
        length_of_audio_seconds: float,
        midi_feature_names: List[str]
) -> np.ndarray:
    midi_loader = MidiLoader()

    midi_features = midi_loader.load(
        midi_file_name=path_to_midi_file,
        frame_rate=frame_rate,
        audio_length_seconds=length_of_audio_seconds,
        only_these_features=midi_feature_names)

    audio_features_generator = HeuristicAudioFeaturesGenerator()
    heuristic_audio_features = audio_features_generator.generate(midi_features)

    features = {
        **heuristic_audio_features,
        **midi_features
    }

    return model.transfer_timbre(features)


def _write_audio_to_file(
        audio: np.ndarray,
        file_path: str,
        sample_rate: int,
        normalize: bool):
    if normalize:
        audio = librosa.util.normalize(audio)
    wavfile.write(file_path, sample_rate, audio)


if __name__ == "__main__":
    main()
