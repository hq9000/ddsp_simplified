import os
from os import listdir
from os.path import join, isfile
from typing import Dict, List

import numpy as np

from tensorflow.keras.callbacks import Callback

import wandb
from ddsp_simplified.config_key_constants import INFERENCE, PATHS_TO_MIDI_FILES_FOR_SYNTHESIS, DATA, FRAME_RATE, \
    MIDI_FEATURES, SAMPLE_RATE, EXAMPLE_MIDI_FILES_DIR
from ddsp_simplified.models import SupervisedAutoencoder
from ddsp_simplified.synthesize_from_midi_lib import synthesize_audio_by_midi


class ModelCheckpoint(Callback):
    def __init__(self, save_dir, monitor, **kwargs):
        super().__init__(**kwargs)

        self.monitor = monitor
        
        self.save_dir = save_dir # wandb/run_name/files/run_name
        self.best_model_dir = os.path.join(save_dir, 'best_model')
        self.final_model_dir = os.path.join(save_dir, 'train_end')
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.final_model_dir, exist_ok=True)
        self.save_path = os.path.join(self.best_model_dir, 'model.ckpt')

        self.best = np.Inf
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.less(current, self.best):
            self.best = current
            self.model.save_weights(self.save_path)
            with open(os.path.join(self.best_model_dir, 'model_info.txt'), 'w') as outfile:
                outfile.write('epoch: {}'.format(epoch))
                for k, v in logs.items():
                    outfile.write('\n{}: {}'.format(k, v))

    # Save also the last model
    def on_train_end(self, logs=None):
        self.model.save_weights(os.path.join(self.final_model_dir, 'model.ckpt'))
        with open(os.path.join(self.final_model_dir, 'model_info.txt'), 'w') as outfile:   
            for k, v in logs.items():
                outfile.write('{}: {}\n'.format(k, v))              

class CustomWandbCallback(Callback):

    def __init__(self, config, **kwargs):
        super().__init__()

        wandb.login()
        wandb.init(project=config['wandb']['project_name'],
                    entity='hq9000',
                    name=config['run_name'],
                    config=config)

        self._config: Dict = config

        self.wandb_run_dir = wandb.run.dir
        
    def on_epoch_end(self, epoch, logs=None):

        data_to_log = dict(logs)

        if epoch % 20 == 0:
            audios = self._generate_audios(self.model, self._config)
            data_to_log["audio_examples"] = {}

            for file_name, audio in audios.items():
                data_to_log["audio_examples"][file_name] = wandb.Audio(audio * 0.4, caption=file_name,
                                                                       sample_rate=self._config[DATA][SAMPLE_RATE])

        wandb.log(data_to_log)

    def _generate_audios(self, model: SupervisedAutoencoder, config: Dict) -> Dict[str, np.ndarray]:

        midi_files_directory: str = config[INFERENCE][EXAMPLE_MIDI_FILES_DIR]
        all_midi_file_names = self._find_all_midi_files_name(midi_files_directory)

        res = {}

        for midi_file_name in all_midi_file_names:
            audio = synthesize_audio_by_midi(model=model,
                                             path_to_midi_file=join(midi_files_directory, midi_file_name),
                                             frame_rate=config[DATA][FRAME_RATE],
                                             length_of_audio_seconds=4.0,
                                             midi_feature_names=config[DATA][MIDI_FEATURES])

            if audio is not None:
                res[midi_file_name] = audio

        return res

    def _find_all_midi_files_name(self, directory: str) -> List[str]:
        return [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.mid')]
