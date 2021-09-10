import numpy as np

from dsp_utils.spectral_ops import compute_loudness, compute_f0, compute_mfcc, compute_logmel
from feature_names import INPUT_FEATURE_LOUDNESS_DB, INPUT_FEATURE_F0_HZ, INPUT_FEATURE_MFCC, INPUT_FEATURE_LOG_MEL

from utilities import concat_dct, frame_generator

## -------------------------------------------------- Feature Extraction ---------------------------------------


def feature_extractor(audio, sample_rate=16000, model=None, frame_rate=250,
                    f0=True, loudness=True, mfcc=False, log_mel=False,
                    mfcc_nfft=1024, l_nfft=2048, logmel_nfft=2048,
                    conf_threshold=0.0):
    """
    mfcc_nfft should be determined by preprocessing timesteps.
    l_nfft and log_mel nfft are used as here in the library."""

    features = {'audio': audio}

    if f0:
        f0, confidence = compute_f0(audio, sample_rate, frame_rate, viterbi=True) 
        f0 = confidence_filter(f0, confidence, conf_threshold)
        features[INPUT_FEATURE_F0_HZ] = f0

    if mfcc:
        # overlap and fft_size taken from the code
        # overlap is the same except for frame size 63
        features[INPUT_FEATURE_MFCC] = compute_mfcc(audio,
                                        fft_size=mfcc_nfft,
                                        overlap=0.75,
                                        mel_bins=128,
                                        mfcc_bins=30)

    if log_mel:
        features[INPUT_FEATURE_LOG_MEL] = compute_logmel(audio,
                                            bins=229, #64
                                            fft_size=logmel_nfft,
                                            overlap=0.75,
                                            pad_end=True,
                                            sample_rate=sample_rate)
                                                                             
    if loudness:                                            
        # apply reverb before l extraction to match
        # room acoustics for timbre transfer
        if model is not None and model.add_reverb: 
            audio = model.reverb({"audio_synth":audio[np.newaxis,:]})[0]

        features[INPUT_FEATURE_LOUDNESS_DB] = compute_loudness(audio,
                                                    sample_rate=sample_rate,
                                                    frame_rate=frame_rate,
                                                    n_fft=l_nfft,
                                                    use_tf=False)                             
    return features

def extract_features_from_frames(frames, **kwargs):
    """Extracts features from multiple frames and concatenates them."""
    return concat_dct([feature_extractor(frame, **kwargs) for frame in frames])    

def process_track(track, sample_rate=16000, audio_length=60, frame_size=64000, **kwargs):
    """Generates frames from a track and extracts features for each frame."""
    MAX_AUDIO_LENGTH = sample_rate*audio_length
    if len(track) > MAX_AUDIO_LENGTH: # trim from the end
        track = track[:MAX_AUDIO_LENGTH]
    frames = frame_generator(track, frame_size) # large chunks of audio
    return extract_features_from_frames(frames, **kwargs)

def confidence_filter(F0, confidence, threshold):
    """
    Silences the time instants where the model confidence is below the given threshold.
    """

    return [f if confidence[idx] >= threshold else 0.0 for idx, f in enumerate(F0)]    