from abc import abstractmethod, ABC
from typing import Optional, List

from tensorflow.keras.models import Model
from tensorflow.keras import metrics as tfkm

from decoders import HARMONIC_OUT_ADDITIONAL
from synthesizers import *


class Autoencoder(Model, ABC):
    def __init__(self,
               preprocessor=None,
               add_reverb=False,
               loss_fn=None,
               n_samples=64000,
               sample_rate=16000,
               tracker_names: Optional[List] = None,
               metric_fns: Optional[Dict] = None,
               additional_harmonic_needed: bool = False,
               **kwargs):

        if tracker_names is None:
            tracker_names = ["spec_loss"]

        if metric_fns is None:
            metric_fns = {}

        super().__init__(**kwargs)
        self.preprocessor = preprocessor
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.loss_fn = loss_fn

        self.harmonic: HarmonicSynthesizer = HarmonicSynthesizer(n_samples=self.n_samples,
                                            sample_rate=self.sample_rate,
                                            name='harmonic',
                                            f0_ratio=1.0)

        self.harmonic_additional: Optional[HarmonicSynthesizer] = None

        if additional_harmonic_needed:
            self.harmonic_additional = HarmonicSynthesizer(
                n_samples=self.n_samples,
                sample_rate=self.sample_rate,
                name='harmonic_additional',
                f0_ratio=1.5)

        self.noise = FilteredNoiseSynthesizer(window_size=0,
                                      initial_bias=-10.0,
                                      name='noise')
        
        self.add_reverb = add_reverb
        if self.add_reverb:
            self.reverb = Reverb(reverb_length=n_samples)
        self.trackers = TrackerGroup(*tracker_names)
        self.metric_fns = metric_fns

    @abstractmethod
    def encode(self, features):
        raise NotImplementedError

    @abstractmethod
    def decode(self, features):
        raise NotImplementedError
    
    def dsp_process(self, features):
        """Synthesizes audio and adds reverb if specified."""

        feature_key_harmonic = "harmonic"
        feature_key_harmonic_additional = "harmonic_additional"
        feature_key_noise = "noise"

        features[feature_key_harmonic] = self.harmonic(features, HARMONIC_OUT)

        if self.harmonic_additional is not None:
            features[feature_key_harmonic_additional] = self.harmonic_additional(features,
                                                                                 HARMONIC_OUT_ADDITIONAL)

        features[feature_key_noise] = self.noise(features)
        outputs = {
            "inputs": features,
            AUDIO_SYNTH: features[feature_key_harmonic] + features[feature_key_noise]
        }

        if self.harmonic_additional is not None:
            outputs[AUDIO_SYNTH] = outputs[AUDIO_SYNTH] + features[feature_key_harmonic_additional]

        if self.add_reverb:
            outputs[AUDIO_SYNTH] = self.reverb(outputs)
        return outputs

    # code from github repo, kept it but unnecessary
    def get_audio_from_outputs(self, outputs):
        """Extract audio output tensor from outputs dict of call()."""
        return outputs['audio_synth']

    def transfer_timbre(self, features):
        model_output = self(features)
        audio_synth = self.get_audio_from_outputs(model_output)
        return audio_synth.numpy().reshape(-1)    

    # is copying necessary?
    #ll(self, features):
    #    _features = features.copy()
    #    _features = self.encode(_features)
    #    _features = self.decode(_features)
    #    outputs = self.dsp_process(_features)       
    #    return outputs

    def call(self, features):
        features = self.encode(features)
        features = self.decode(features)
        outputs = self.dsp_process(features)       
        return outputs        
  
    @tf.function
    def train_step(self, x):
        """Run the core of the network, get predictions and loss."""

        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            loss = self.loss_fn({'audio': x_pred[AUDIO_SYNTH], 'target_audio':x["audio"]})
            total_loss = loss["total_loss"] if "total_loss" in loss else loss["spec_loss"]
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)      
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        metrics = {name:fn(x, x_pred) for name, fn in self.metric_fns.items()}
        self.trackers.update_state(loss)
        self.trackers.update_state(metrics)
        return self.trackers.result()
    
    @tf.function
    def test_step(self, x):
        x_pred = self(x,training=False)
        loss = self.loss_fn({'audio': x_pred[AUDIO_SYNTH], 'target_audio':x["audio"]})
        metrics = {name:fn(x, x_pred) for name, fn in self.metric_fns.items()}
        self.trackers.update_state(loss)
        self.trackers.update_state(metrics)
        return self.trackers.result()

    @property
    def metrics(self):
        return self.trackers.trackers.values()
  
class SupervisedAutoencoder(Autoencoder):
    def __init__(self,
               preprocessor=None,
               encoder=None,
               decoder=None,
               add_reverb=False,
               loss_fn=None,
               n_samples=64000,
               sample_rate=16000,
               tracker_names=["spec_loss"],
               metric_fns={},
               additional_harmonic_needed: bool = False,
               **kwargs):
        
        super().__init__(preprocessor, add_reverb, loss_fn, n_samples, sample_rate,
                         tracker_names=tracker_names, metric_fns=metric_fns, **kwargs,
                         additional_harmonic_needed=additional_harmonic_needed)
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, features): 
        """Loudness and F0 is read. z is encoded optionally."""

        if self.preprocessor is not None:  # Downsample and Scale the features
            processed_features = self.preprocessor(features)
            features.update(processed_features)
        if self.encoder is not None:
            outputs = self.encoder(features)
            features.update(outputs) 
        return features
    
    def decode(self, features):
        """Map the f,l (,z) parameters to synthesizer parameters."""
        
        outputs = self.decoder(features)
        features.update(outputs)
        return features
     
class UnsupervisedAutoencoder(Autoencoder):
    def __init__(self,
               encoder,
               decoder,    
               preprocessor=None,
               add_reverb=False,
               loss_fn=None,
               n_samples=64000,
               sample_rate=16000,
               tracker_names=["spec_loss"],
               metric_fns={},
               **kwargs):
        
        super().__init__(preprocessor, add_reverb, loss_fn, n_samples, sample_rate, tracker_names=tracker_names, metric_fns=metric_fns, **kwargs)
        
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, features):
        if self.preprocessor is not None:
            features.update(self.preprocessor(features))
        outputs = self.encoder(features)
        features.update(outputs)
        return features
    
    def decode(self, features):  
        """Map the f, l (,z) parameters to synthesizer parameters."""
        output = self.decoder(features)   
        features.update(output)
        return features

class TrackerGroup():
    def __init__(self,*names):
        self.trackers = {name:tfkm.Mean(name+"_tracker") for name in names}

    def update_state(self, dct):
        for k,v in dct.items():
            self.trackers[k].update_state(v)
        
    def result(self):
        return {name:tracker.result() for name,tracker in self.trackers.items()}