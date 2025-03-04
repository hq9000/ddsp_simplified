import os
import pathlib
from typing import Dict, Tuple

import keras.callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from config_key_constants import MODEL, ADDITIONAL_HARMONIC_NEEDED
from preprocessing import F0LoudnessAndMidiFeaturesPreprocessor, LoudnessPreprocessor
from encoders import SupervisedEncoder, UnsupervisedEncoder
from models import SupervisedAutoencoder, UnsupervisedAutoencoder
from decoders import DecoderWithoutLatent, DecoderWithLatent
from losses import SpectralLoss, MultiLoss
from callbacks import ModelCheckpoint, CustomWandbCallback 

from metrics import f0_midi_scaled_L1_loss #????

from dataloader import make_supervised_dataset, make_unsupervised_dataset

# -------------------------------------- Models -------------------------------------------------

# TODO: enc, dec params
# TODO metric fns
# preprocessor ? 
def make_unsupervised_model(config):

    preprocessor = LoudnessPreprocessor(timesteps=config['data']['preprocessing_time'])

    encoder = UnsupervisedEncoder(timesteps=config['data']['preprocessing_time']) 
    decoder = DecoderWithLatent(timesteps=config['model']['decoder_time'])
    
    loss = SpectralLoss() if config['loss']['type'] == 'spectral' else MultiLoss()
    if loss.name== 'spectral_loss':
        tracker_names = ['spec_loss']
    else:
        tracker_names = ['spec_loss', 'perc_loss', 'total_loss']
    metric_fns = {"F0_recons_L1": f0_midi_scaled_L1_loss}        
    model = UnsupervisedAutoencoder(
                                encoder=encoder,
                                decoder=decoder,
                                preprocessor=preprocessor,
                                loss_fn=loss,
                                tracker_names=tracker_names,
                                metric_fns=metric_fns,
                                add_reverb=config['model']['reverb'])
    return model


def make_supervised_model(config) -> SupervisedAutoencoder:
    """Creates the necessary components of a supervised ddsp using the config."""
    preprocessor = F0LoudnessAndMidiFeaturesPreprocessor(timesteps=config['data']['preprocessing_time'])

    additional_harmonic_needed = False
    if ADDITIONAL_HARMONIC_NEEDED in config[MODEL]:
        additional_harmonic_needed = additional_harmonic_needed = config[MODEL][ADDITIONAL_HARMONIC_NEEDED]

    if config['model']['encoder']:
        encoder = SupervisedEncoder()
        decoder = DecoderWithLatent(timesteps=config['model']['decoder_time'])
    else:
        encoder = None
        decoder = DecoderWithoutLatent(timesteps=config['model']['decoder_time'],
                                       midi_features=config['data']['midi_features'],
                                       additional_harmonic_needed=additional_harmonic_needed)

    loss = SpectralLoss(logmag_weight=config['loss']['logmag_weight'])

    model = SupervisedAutoencoder(preprocessor=preprocessor,
                                encoder=encoder,
                                decoder=decoder,
                                loss_fn=loss,
                                tracker_names=['spec_loss'],
                                add_reverb=config['model']['reverb'],
                                additional_harmonic_needed=additional_harmonic_needed)
    return model

# -------------------------------------- Optimizer -------------------------------------------------


def make_optimizer(config) -> Adam:

    if 'decay_steps' in config['optimizer']:
        learning_rate = ExponentialDecay(initial_learning_rate=config['optimizer']['lr'],
                                         decay_steps=config['optimizer']['decay_steps'],
                                         decay_rate=config['optimizer']['decay_rate'])
    else:
        learning_rate = config['optimizer']['lr']

    optimizer = Adam(learning_rate=learning_rate)

    return optimizer                                    

# -------------------------------------- Callbacks -------------------------------------------------

def create_callbacks(config, monitor):
    # It looks ugly, but is necessary
    if config['model']['dir']: # if dir specified, save there
        model_dir = config['model']['dir']
        if not config['wandb']:
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor)]
        else:
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor),
                        CustomWandbCallback(config)]
    else:
        if not config['wandb']: # define a save_dir
            model_dir = "model_checkpoints/{}".format(config['run_name'])
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor)]
        else:  # save to wandb.run.dir
            log_dir = str(pathlib.Path(__file__).parent.resolve()) + '/tensorboard'
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir)
            wandb_callback = CustomWandbCallback(config)
            model_dir = os.path.join(wandb_callback.wandb_run_dir, config['run_name'])
            callbacks = [ModelCheckpoint(save_dir=model_dir, monitor=monitor),
                         wandb_callback,
                         tensorboard_callback]
    return callbacks


# -------------------------------------- Datasets -------------------------------------------------      
def make_supervised_dataset_from_config(config: Dict):
    pitch_shifts = tuple(range(
        config['data']['augmentation']['pitch_shift']['start'],
        config['data']['augmentation']['pitch_shift']['end'] + 1,
    ))
    return make_supervised_dataset(config['data']['path'],
                                mfcc=config['model']['encoder'],
                                mfcc_nfft=config['data'].get('mfcc_nfft', 1024),
                                batch_size=config['training']['batch_size'],
                                sample_rate=config['data']['sample_rate'],
                                normalize=config['data']['normalize'],
                                conf_threshold=config['data']['confidence_threshold'],
                                frame_rate=config['data'].get('frame_rate', 250),
                                midi_feature_names=config['data'].get('midi_features', None),
                                pitch_shifts=pitch_shifts,
                                silence_tail_seconds=config['data']['compression']['silence_tail_size_seconds']
                                   )


def make_unsupervised_dataset_from_config(config):
    return make_unsupervised_dataset(config['data']['path'],
                                batch_size=config['training']['batch_size'],
                                sample_rate=config['data']['sample_rate'],
                                normalize=config['data']['normalize'],
                                frame_rate=config['data']['preprocessing_time'])