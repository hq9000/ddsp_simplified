from typing import Dict, Tuple

import numpy as np
from tensorflow.keras import layers as tfkl

from dsp_utils import spectral_ops
from dsp_utils.core import resample, midi_to_hz, hz_to_midi
from feature_names import FEATURE_F0_HZ, FEATURE_F0_MIDI_SCALED, FEATURE_LD_SCALED, INPUT_FEATURE_F0_HZ, \
    INPUT_FEATURE_LOUDNESS_DB

F0_RANGE = spectral_ops.F0_RANGE
LD_RANGE = spectral_ops.LD_RANGE
CC_RANGE = 128
VELOCITY_RANGE = 128
PITCH_RANGE = 128

from utilities import at_least_3d

class F0LoudnessAndMidiFeaturesPreprocessor(tfkl.Layer):
    """Resamples and scales 'f0_hz' and 'loudness_db' features. Used in the Supervised Setting."""

    def __init__(self, timesteps=250, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs: Dict[str, np.ndarray], *args, **kwargs):

        preprocessed_audio_features = self._preprocess_audio_features(inputs={
            INPUT_FEATURE_F0_HZ: inputs[INPUT_FEATURE_F0_HZ],
            INPUT_FEATURE_LOUDNESS_DB: inputs[INPUT_FEATURE_LOUDNESS_DB]
        })

        inputs_with_midi_features_only = dict(inputs)  # create a copy

        del inputs_with_midi_features_only[INPUT_FEATURE_F0_HZ]
        del inputs_with_midi_features_only[INPUT_FEATURE_LOUDNESS_DB]

        preprocessed_midi_features = self._preprocess_midi_features(
            inputs=inputs_with_midi_features_only
        )

        return {
            **preprocessed_audio_features,
            **preprocessed_midi_features
        }

    def _preprocess_audio_features(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        # Downsample features, but do not update them in the dict
        f0_hz = self.resample(inputs[INPUT_FEATURE_F0_HZ])
        loudness_db = self.resample(inputs[INPUT_FEATURE_LOUDNESS_DB])

        # For NN training, scale frequency and loudness to the range [0, 1].
        # Log-scale f0 features. Loudness from [-1, 0] to [1, 0].

        f0_midi_scaled = hz_to_midi(f0_hz) / F0_RANGE
        ld_scaled = (loudness_db / LD_RANGE) + 1.0

        return {FEATURE_F0_HZ: at_least_3d(inputs[INPUT_FEATURE_F0_HZ]),  # kept for the harmonic synth, convert to 3d here
                FEATURE_F0_MIDI_SCALED: f0_midi_scaled,  # used in the decoder in this form
                FEATURE_LD_SCALED: ld_scaled}  # same as f0_midi_scaled

    def _preprocess_midi_features(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        for (feature_name, data) in inputs.items():
            resampled = self.resample(data)
            min_value, max_value = self._get_range_of_midi_feature(feature_name)
            scaled = (resampled - min_value) / (max_value - min_value)
            res[feature_name] = scaled

        return res

    def _get_range_of_midi_feature(self, feature_name: str) -> Tuple[float, float]:
        return 0.0, 128.0

    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method="linear")        

# TODO: fix the Encoder_f
# Downsample the f and l using the F0LoudnessAndMidiFeaturesPreprocessor
# Remove this class
class LoudnessPreprocessor(tfkl.Layer):

    def __init__(self, timesteps=250, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):

        loudness_db = inputs["loudness_db"]

        # Resample features to time_steps.
        loudness_db = self.resample(loudness_db)

        # For NN training, scale frequency and loudness to the range [0, 1].
        ld_scaled = (loudness_db / LD_RANGE) + 1.0

        return {"ld_scaled":ld_scaled}
        
    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method="linear")


# TODO: delete??
class MidiF0LoudnessPreprocessor(tfkl.Layer):
    """Scales the loudness, converts scaled midi to hz and resamples. Used in the Unsupervised setting."""

    def __init__(self, timesteps=1000, **kwargs):
        super().__init__(**kwargs)
        self.timesteps = timesteps

    def call(self, inputs):
       
        loudness_db, f0_scaled = inputs["loudness_db"], inputs["f0_midi_scaled"]
               
        # Resample features to time_steps.
        f0_scaled = resample(f0_scaled, self.timesteps)
        loudness_db = resample(loudness_db, self.timesteps)
        
        # For NN training, scale frequency and loudness to the range [0, 1].
        ld_scaled = (loudness_db / LD_RANGE) + 1.0
        
        # ???????????????????????????
        # Convert scaled midi to hz for the synthesizer
        f0_hz = midi_to_hz(f0_scaled*F0_RANGE)
        
        f0_hz = resample(at_least_3d(f0_hz), 1000)
       
        return {"f0_hz":f0_hz, "loudness_db":loudness_db, "f0_midi_scaled":f0_scaled, "ld_scaled":ld_scaled}

    def resample(self, x):
        x = at_least_3d(x)
        return resample(x, self.timesteps, method="linear") 