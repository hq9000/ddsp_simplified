import glob
import os.path
from hashlib import md5
from typing import Dict, Optional, List, Tuple

import numpy as np

from tensorflow.data import Dataset
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_dataset_ops import TensorSliceDataset

from ddsp_simplified.dsp_utils.spectral_ops import compute_loudness
from ddsp_simplified.synthesize_from_midi_lib import get_midi_feature_names_augmented_with_pitch_velocity_sequence_number
from feature_extraction import extract_features_from_frames, feature_extractor, \
    extract_features_from_audio_frames_using_heuristic_generator
from feature_names import MIDI_FEATURE_PITCH, MIDI_FEATURE_VELOCITY, MIDI_FEATURE_NOTE_SEQUENCE_NUMBER
from utilities import frame_generator, load_track, get_raw_midi_features_from_file, generate_midi_features_examples, concat_dct
from utils.cache import Cache
from utils.dataset_compressor import DatasetCompressor

MIDI_FILE_EXTENSION = 'MID'


def _make_dataset(features: Dict[str, np.ndarray], batch_size: int = 32, seed: str = None) -> TensorSliceDataset:
    dataset: TensorSliceDataset = Dataset.from_tensor_slices(features)
    dataset = dataset.shuffle(len(features)*2, seed, True)  # shuffle at each iteration
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)  # prefetch 1 batch

    return dataset

# -------------------------------------------- Supervised Dataset -------------------------------------------------

def guess_midi_file_name_by_audio_file_name(audio_file_name: str) -> str:
    return os.path.splitext(audio_file_name)[0] + '.' + MIDI_FILE_EXTENSION


def _filter_midi_features_data(raw_midi_features_data: Dict[str, np.array], feature_names_to_leave: List[str]) -> Dict[str, np.array]:
    """
    Get a dict of feature names and return its version with only requested features left.
    """
    return {feature_name: raw_midi_features_data[feature_name] for feature_name in feature_names_to_leave}


def _apply_pitch_shift_to_midi_features(midi_features: Dict[str, np.ndarray], pitch_shift: int):
    pitches = midi_features[MIDI_FEATURE_PITCH]
    non_zero_indices = pitches > 0.000001
    pitches[non_zero_indices] = pitches[non_zero_indices] + pitch_shift


def _generate_loudness_based_velocities(
        audio_file_name: str,
        pitch_shift: int,
        note_sequence_numbers: np.ndarray,
        audio_data: np.ndarray,
        frame_rate: int,
        sample_rate: int) -> np.ndarray:

    loudness_contour = _generate_loudness_contour(audio_data, audio_file_name, frame_rate, pitch_shift, sample_rate)

    note_peak_loudness_map = _generate_note_peak_loudness_map(note_sequence_numbers, loudness_contour)

    peak_loudnesses = np.ndarray(note_peak_loudness_map.values())

    peak_loudness_std = np.std(peak_loudnesses)
    peak_loudness_mean = np.mean(peak_loudnesses)

    res = np.zeros(shape=note_sequence_numbers.shape)

    reference_dataset_velocity_std = 19.044096499689005
    reference_dataset_velocity_mean = 65.02882818182805

    for i, note_sequence_number in enumerate(note_sequence_numbers):
        note_peak_loudness = note_peak_loudness_map[note_sequence_number]
        velocity = (note_peak_loudness - peak_loudness_mean) / peak_loudness_std * reference_dataset_velocity_std + reference_dataset_velocity_mean
        res[i] = velocity

    return res

def _generate_note_peak_loudness_map(note_sequence_numbers: np.ndarray, loudness_contour: np.ndarray) -> Dict[float, float]:
    pass

def _generate_loudness_contour(audio_data: np.ndarray, audio_file_name: str, frame_rate: int, pitch_shift: int, sample_rate: int) -> np.ndarray:
    cache = Cache.get_instance()
    cache_key = '-'.join([
        md5(audio_file_name.encode('utf-8')).hexdigest(),
        'pitch_shift=' + str(pitch_shift),
        'sr=' + str(sample_rate),
        'fr=' + str(frame_rate),
        'loudness_contour'
    ])
    if not cache.has_numpy_array(cache_key):
        loudness_contour = compute_loudness(audio=audio_data, sample_rate=sample_rate, frame_rate=frame_rate)
        cache.put_numpy_array(cache_key, loudness_contour)
    loudness_contour = cache.get_numpy_array(cache_key)
    # loudness is known to be behind the waveform by approximately this
    # many frames. What we need is to make loudness picks fall into the midi
    # note region, to be on the safe side, we shift it slightly.
    loudness_shift_frames = 25
    shifted_loudness_contour = np.concatenate(
        (np.zeros(loudness_shift_frames), loudness_contour[:-loudness_shift_frames]), axis=0)
    return shifted_loudness_contour


def make_supervised_dataset(path, mfcc=False, batch_size=32, sample_rate=16000,
                            normalize=False, conf_threshold=0.0, mfcc_nfft=1024,
                            frame_rate: int = 250,
                            midi_feature_names: Optional[List[str]] = None,
                            empty_list_to_put_train_feature_frames: Optional[List] = None,
                            empty_list_to_put_val_feature_frames: Optional[List] = None,
                            pitch_shifts: Tuple[int, ...] = (0, ),
                            silence_tail_seconds: float = 0.2,
                            correct_velocities_using_loudness: bool = False
                            ) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """Loads all the mp3 and (optionally) midi files in the path, creates frames and extracts features."""

    ordered_audio_frames = []
    ordered_midi_features_frames = []

    length_of_example_seconds = 4.0

    # this one is essentially just syntactic sugar
    need_midi = True if midi_feature_names is not None else False

    KEY_AUDIO = 'audio'
    KEY_MIDI = 'midi'

    for audio_file_name in glob.glob(path+'/*.mp3'):
        for pitch_shift in pitch_shifts:
            audio_data = load_track(
                audio_file_name,
                sample_rate=sample_rate,
                normalize=normalize,
                pitch_shift=pitch_shift
            )

            midi_file_name = guess_midi_file_name_by_audio_file_name(audio_file_name)
            midi_feature_names_with_pitch_and_velocity_and_seq_number = get_midi_feature_names_augmented_with_pitch_velocity_sequence_number(
                midi_feature_names
            )

            raw_midi_features_data_with_pitch_and_velocity = get_raw_midi_features_from_file(
                path_to_midi_file=midi_file_name,
                frame_rate=frame_rate,
                audio_length_seconds=audio_data.shape[0] / sample_rate,
                only_these_features=midi_feature_names_with_pitch_and_velocity_and_seq_number
            )

            if True:
                corrected_velocities = _generate_loudness_based_velocities(
                    audio_file_name=audio_file_name,
                    pitch_shift=pitch_shift,
                    note_sequence_numbers=raw_midi_features_data_with_pitch_and_velocity[MIDI_FEATURE_NOTE_SEQUENCE_NUMBER],
                    audio_data=audio_data,
                    frame_rate=frame_rate,
                    sample_rate=sample_rate
                )

                raw_midi_features_data_with_pitch_and_velocity[MIDI_FEATURE_VELOCITY] = corrected_velocities

            data_compressor = DatasetCompressor()

            compressed_audio_data, compressed_raw_midi_features_data_with_pitch_and_velocity = \
                data_compressor.compress_data(
                    audio_data,
                    raw_midi_features_data_with_pitch_and_velocity,
                    frame_rate=frame_rate,
                    sample_rate=sample_rate,
                    silence_tail_size_frames=int(silence_tail_seconds * frame_rate)
                )

            _apply_pitch_shift_to_midi_features(compressed_raw_midi_features_data_with_pitch_and_velocity, pitch_shift)

            generated_midi_feature_examples = generate_midi_features_examples(
                compressed_raw_midi_features_data_with_pitch_and_velocity,
                int(length_of_example_seconds * frame_rate)
            )

            generated_audio_frames = frame_generator(compressed_audio_data, int(length_of_example_seconds * sample_rate))

            ordered_midi_features_frames.extend(generated_midi_feature_examples)
            ordered_audio_frames.extend(generated_audio_frames)

    combined_frames = []
    if need_midi:
        assert len(ordered_audio_frames) == len(ordered_midi_features_frames)
        for (audio_frame, midi_features_frame) in zip(ordered_audio_frames, ordered_midi_features_frames):
            combined_frames.append({
                KEY_AUDIO: audio_frame,
                KEY_MIDI: midi_features_frame
            })
    else:
        for audio_frame in ordered_audio_frames:
            combined_frames.append({
                KEY_AUDIO: audio_frame,
            })

    trainX, valX = train_test_split(combined_frames)

    train_shuffled_audio_frames = [x[KEY_AUDIO] for x in trainX]
    val_shuffled_audio_frames = [x[KEY_AUDIO] for x in valX]

    # audio_and_midi_features_frames = np.concatenate(audio_and_midi_features_frames, axis=0)

    print('Train set size: {}\nVal set size: {}'.format(len(trainX),len(valX)))

    extract_f0_and_loudness_from_audio = False

    if extract_f0_and_loudness_from_audio:
        train_audio_and_audio_features = extract_features_from_frames(train_shuffled_audio_frames, mfcc=mfcc, sample_rate=sample_rate,
                                                                      conf_threshold=conf_threshold, mfcc_nfft=mfcc_nfft)
        val_audio_and_audio_features = extract_features_from_frames(val_shuffled_audio_frames, mfcc=mfcc, sample_rate=sample_rate,
                                                                    conf_threshold=conf_threshold, mfcc_nfft=mfcc_nfft)
    else:
        train_audio_and_audio_features = extract_features_from_audio_frames_using_heuristic_generator(
            audio_frames=train_shuffled_audio_frames,
            midi_frames=trainX
        )
        val_audio_and_audio_features = extract_features_from_audio_frames_using_heuristic_generator(
            audio_frames=val_shuffled_audio_frames,
            midi_frames=valX
        )

    if need_midi:
        train_shuffled_midi_frames = [x[KEY_MIDI] for x in trainX]
        val_shuffled_midi_frames = [x[KEY_MIDI] for x in valX]

        train_shuffled_midi_frames = concat_dct(train_shuffled_midi_frames)
        val_shuffled_midi_frames = concat_dct(val_shuffled_midi_frames)

        combined_train_features = {
            **train_audio_and_audio_features,
            **train_shuffled_midi_frames
        }

        combined_val_features = {
            **val_audio_and_audio_features,
            **val_shuffled_midi_frames
        }
    else:
        combined_train_features = train_audio_and_audio_features
        combined_val_features = val_audio_and_audio_features

    if empty_list_to_put_train_feature_frames is not None:
        empty_list_to_put_train_feature_frames.append(combined_train_features)

    if empty_list_to_put_val_feature_frames is not None:
        empty_list_to_put_val_feature_frames.append(combined_val_features)

    train_dataset = _make_dataset(combined_train_features, batch_size)
    val_dataset = _make_dataset(combined_val_features, batch_size)

    return train_dataset, val_dataset, None

# -------------------------------------------- Unsupervised Datasets ----------------------------------------------

def make_unsupervised_dataset(path, batch_size=32, sample_rate=16000, normalize=False, frame_rate=250):
    frames = []
    for file in glob.glob(path+'/*.mp3'):    
        track = load_audio_track(file, sample_rate=sample_rate, normalize=normalize)
        frames.append(frame_generator(track, 4*sample_rate)) # create 4 seconds long frames     
    frames = np.concatenate(frames, axis=0)   
    trainX, valX = train_test_split(frames)
    train_features = extract_features_from_frames(trainX, f0=False, mfcc=True, log_mel=True,
                                                sample_rate=sample_rate, frame_rate=frame_rate)
    val_features = extract_features_from_frames(valX, f0=False, mfcc=True, log_mel=True,
                                                sample_rate=sample_rate, frame_rate=frame_rate)
    return _make_dataset(train_features, batch_size), _make_dataset(val_features, batch_size), None    

#/scratch/users/hbalim15/tensorflow_datasets/nsynth/
def make_nsynth_dataset(batch_size, path='nsynth/gansynth_subset.f0_and_loudness:2.3.3'):
    split=["train[:80%]", 'validation[:10%]', 'test[:10%]']
    train_set, val_set, test_set = tfds.load(path,
                                            download=False,
                                            shuffle_files=True,
                                            batch_size=batch_size,
                                            split=split).prefetch(1).map(lambda x: preprocess_ex(x))
    return train_set, val_set, test_set

def preprocess_ex(ex):   
    dct = {'pitch': ex['pitch'],  
          'f0_hz': ex['f0']['hz'],
          'loudness_db': ex['loudness']['db']}
    dct.update(feature_extractor(ex['audio'], sample_rate=16000, frame_rate=250,
                        f0=False, mfcc=True, log_mel=True))
    return dct  


#import tensorflow as tf

#from tensorflow.keras.utils import Sequence

#def make_datasets(batch_size, percent=100):
#    datasets = [create_tf_dataset_from_npzs(glob.glob("data/{}/*.npz".format(folder)), batch_size, percent) for 
#                         folder in ["train","validation","test"]]
#    return datasets

#def create_tf_dataset_from_npzs(npzs, batch_size, percent):
#    def generator(dataset_generator):
#        while True:
#            for batch in dataset_generator:
#                yield batch
#    npzs = npzs[:int(len(npzs)*percent/100)]
#    data_generator = DataGenerator(npzs, batch_size)
#    generator_func = lambda: generator(data_generator)
#    sample = data_generator[0]
#    output_shapes = {k:sample[k].shape for k in sample}
#    output_types = {k:sample[k].dtype for k in sample}
#    dataset = Dataset.from_generator(generator_func, output_shapes=output_shapes, output_types=output_types)
#    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#    dataset.my_len = len(data_generator)
#    return dataset

#class DataGenerator(Sequence):
#    def __init__(self, list_IDs, batch_size=64, shuffle=True, **kwargs):
#        super().__init__(**kwargs)
#        self.batch_size = batch_size
#        self.list_IDs = list_IDs[:-(len(list_IDs)%batch_size)]
#        self.shuffle = shuffle
#        self.reset()
#
#    def __len__(self):
#        'Denotes the number of batches per epoch'
#        return int(np.floor(len(self.list_IDs) / self.batch_size))
#
#    def __getitem__(self, index):
#        'Generate one batch of data'
#        # Generate indexes of the batch
#        if (index+1)*self.batch_size >= len(self.indexes):
#            self.reset()
#        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#
#        # Find list of IDs
#        list_IDs_temp = [self.list_IDs[k] for k in indexes]
#
#        # Generate data
#        X = self.__data_generation(list_IDs_temp)
#
#        return X
#
#    def reset(self):
#        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.list_IDs))
#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)
#
#    def __data_generation(self, list_IDs_temp):
#        batch = {}
#        arrays = [np.load(ID) for ID in list_IDs_temp]
#        for file in arrays[0].files:
#            batch[file] = np.array([arr[file] for arr in arrays])
#        return batch