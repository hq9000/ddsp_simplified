import argparse

import numpy as np
import yaml

from models import SupervisedAutoencoder, Autoencoder
from train_utils import make_supervised_model
from utils.midi_loader import MidiLoader


def main():
    parser = argparse.ArgumentParser(description='Script to synthesize audio using trained model and midi file.')
    parser.add_argument('-c', '--config-path', type=str, required=True, help='Path to config file.')
    parser.add_argument('-m', '--midi-file', type=str, default='', help='Path to the input midi file.')
    parser.add_argument('-o', '--output-dir', type=str, default='', help='Output audio directory.')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = dict(yaml.load(file, Loader=yaml.FullLoader))

    model = _create_model_and_load_weights(config)


def _create_model_and_load_weights(config) -> SupervisedAutoencoder:
    model = make_supervised_model(config)
    model.load_weights(config['model']['path'])
    return model

def _synthesize_audio_by_midi(model: Autoencoder, path_to_midi_file: str, sample_rate=16000) -> np.ndarray:
    midi_loader = MidiLoader()
    midi_features = midi_loader.load(path_to_midi_file)


    track = load_track(path, sample_rate, pitch_shift=pitch_shift, normalize=normalize)
    features = process_track(track, model=model, **kwargs)
    features["loudness_db"] +=  scale_loudness
    transfered_track = model.transfer_timbre(features)
    return transfered_track

if __name__ == "__main__":
    main()
