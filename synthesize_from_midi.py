import argparse
from typing import List

import librosa
import numpy as np
import yaml

from ddsp_simplified.config_key_constants import INFERENCE, SAVED_MODEL_PATH
from ddsp_simplified.synthesize_from_midi_lib import synthesize_audio_by_midi
from models import SupervisedAutoencoder
from train_utils import make_supervised_model
import scipy.io.wavfile as wavfile


def main():
    parser = argparse.ArgumentParser(description='Script to synthesize audio using trained model and midi file.')
    parser.add_argument('-c', '--config-path', type=str, required=True, help='Path to config file.')
    parser.add_argument('-m', '--midi-file', type=str, default='', help='Path to the input midi file.')
    parser.add_argument('-o', '--output-dir', type=str, default='', help='Output audio directory.')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))

    frame_rate: int = config['data']['frame_rate']
    model = _create_model_and_load_weights(config)
    midi_feature_names: List[str] = config['data']['midi_features']

    audio = synthesize_audio_by_midi(model,
                                     path_to_midi_file=args.midi_file,
                                     frame_rate=frame_rate,
                                     length_of_audio_seconds=4,  # important! anything but 4 wont work!!!!
                                     midi_feature_names=midi_feature_names)

    _write_audio_to_file(audio, '/tmp/hello_world1.wav', sample_rate=16000, normalize=False)


def _create_model_and_load_weights(config) -> SupervisedAutoencoder:
    model = make_supervised_model(config)
    model.load_weights(config[INFERENCE][SAVED_MODEL_PATH])
    return model


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
