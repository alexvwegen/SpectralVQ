import librosa
import numpy as np


class AudioPipeline:
    """
        Initializes general audio pipeline with the given parameters.

        Parameters
        ----------
        rate : int
            Sample rate.
        n_fft : int
            FFT size.
        n_mels : int
            Number of mel bins.
        n_mfcc : int
            Number of mfccs.
        fmin : int
            Minimum frequency (Hz) for mel scaling.
        fmax : int
            Maximum frequency (Hz) for mel scaling.
        onset_boundaries : list[int]
            Boundary frequencies (Hz) for multi-band onset detection.
        retrievals : list[str]
            List of features the pipeline should compute. 
            <br> Can contain: <i>audio, linmag, logmag, mel, mfcc, chroma, onset, multionset</i>
    """
    def __init__(self,
                 rate=44100,
                 n_fft=1024,
                 n_mels=128,
                 n_mfcc=20,
                 fmin=0,
                 fmax=10000,
                 onset_boundaries=[0, 200, 500, 1300, 5500, 10000],
                 retrievals=["logmag"]):
        self.rate = rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax
        self.mel_filters = self._get_mel_filters()
        self.onset_channels = self._get_mel_channels(onset_boundaries)
        self.retrievals = retrievals

        self.dependencies = {
            "audio": None,
            "linmag": "audio",
            "logmag": "linmag",
            "mel": "logmag",
            "mfcc": "mel",
            "chroma": "linmag",
            "onset": "mel",
            "multionset": "mel"
        }

        self.feature_funcs = {
            "linmag": self.stft,
            "logmag": self._compress,
            "mel": self.mel,
            "chroma": self.chroma,
            "mfcc": self.mfcc,
            "onset": self.onset_envelope,
            "multionset": self.multi_onsets
        }

    def _get_mel_channels(self, hz_channels):
        return [int(librosa.hz_to_mel(hz)) for hz in hz_channels]
    
    def load_audio(self, path):
        """
        Load audio data from file.
        """
        audio, _ = librosa.load(path, sr=self.rate)
        return audio
    
    def stft(self, audio):
        """
        Compute the STFT magnitudes.
        """
        return np.abs(librosa.stft(y=audio, n_fft=self.n_fft))
    
    def _get_mel_filters(self):
        """
        Initialize the mel filter bank.
        """
        return librosa.filters.mel(sr=self.rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
    
    def _compress(self, S):
        """
        Convert a linear spectrogram to log-power.
        """
        return np.log(np.maximum(1e-5, S))
    
    def _decompress(self, S):
        """
        Convert a log-power spectrogram to linear.
        """
        return np.exp(S)

    def mel(self, S):
        """
        Mel-scale log-power magnitudes.
        """
        return self.mel_filters @ S
    
    def mfcc(self, S):
        """
        Compute mel frequency cepstral coefficients from log-power mels.
        """
        return librosa.feature.mfcc(S=S, n_mfcc=self.n_mfcc)
    
    def chroma(self, S):
        """
        Compute the chromagram from magnitudes.
        """
        return librosa.feature.chroma_stft(S=S)
    
    def onset_envelope(self, S):
        """
        Compute the onset strength envelope.
        """
        onset_env = librosa.onset.onset_strength(S=S)
        return onset_env
    
    def multi_onsets(self, S):
        """
        Compute the multi-band and single onset strength envelopes.
        """
        multi_onset = librosa.beat.onset.onset_strength_multi(S=S, channels=self.onset_channels)
        return multi_onset
    
    def _get_onset_mask(self, multi_onsets):
        """
        Max-mask for multi-band onsets.
        """
        max_mask = np.zeros_like(multi_onsets).T
        max_mask[np.arange(max_mask.shape[0]), np.argmax(multi_onsets.T, axis=1)] = 10.0
        return max_mask
    
    def weighted_onset_features(self, S):
        """
        Compute weighted multi-band.
        """
        multi_onsets = self.multi_onsets(S)
        onset_env = self.onset_envelope(S)
        mask = self._get_onset_mask(multi_onsets)
        weighted_onsets = np.expand_dims(onset_env, axis=-1) * mask
        return weighted_onsets
    
    def get_feature(self, feature, required):
        # If the feature is already computed, return it
        if feature in required:
            return required[feature]

        # Find the dependency for this feature
        dependency = self.dependencies[feature]

        # Base case: If no dependency, compute directly
        if dependency is None:
            required[feature] = self.feature_funcs[feature](required.get("audio"))
        else:
            # Resolve the dependency recursively
            dependency_value = self.get_feature(dependency, required)
            # Compute the current feature based on its dependency
            required[feature] = self.feature_funcs[feature](dependency_value)

        return required[feature]

        
    def __call__(self, path):
        # Initialize dictionaries
        retrieved = {feature: None for feature in self.retrievals}
        required = {}

        # Load audio data and add to `required`
        required["audio"] = self.load_audio(path)

        # Resolve each requested feature
        for r in self.retrievals:
            required[r] = self.get_feature(r, required)

        # Prepare the final output
        for wanted in retrieved.keys():
            retrieved[wanted] = required[wanted]

        return retrieved
    
    def vocode_mel(self, M, gl_iters=64):
        magnitudes = librosa.feature.inverse.mel_to_stft(M=M, sr=self.rate, n_fft=self.n_fft)
        audio = self.vocode_magnitudes(magnitudes)
        return audio

    def vocode_magnitudes(self, S, gl_iters=64):
        audio = librosa.griffinlim(S=S, n_fft=self.n_fft, n_iter=gl_iters)
        return audio

    @classmethod
    def from_config(cls, configs):
        return cls(**configs)