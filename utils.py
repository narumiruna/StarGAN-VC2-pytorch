import librosa
import numpy as np
import pyworld


def load_audio(path, sr=None, normalize=True, trim=True):
    y, _ = librosa.load(path, sr=sr)
    if normalize:
        y = librosa.util.normalize(y)
    if trim:
        y, _ = librosa.effects.trim(y)
    return y


def world_decompose(y, sr, num_dims=36):
    y = y.astype(np.float64)

    f0, t = pyworld.harvest(y, sr)
    sp = pyworld.cheaptrick(y, f0, t, sr)
    ap = pyworld.d4c(y, f0, t, sr)
    code = pyworld.code_spectral_envelope(sp, sr, num_dims)
    return f0, sp, ap, code
