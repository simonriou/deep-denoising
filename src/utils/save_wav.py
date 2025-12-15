import torch
import scipy.io.wavfile as wavfile
import numpy as np

"""
This function saves a waveform to a .wav file from a PyTorch tensor or Numpy array.
It is used for debugging and verification purposes.
"""

import numpy as np
from scipy.io import wavfile

def save_wav(data, path, sample_rate=16000):
    """
    Saves a waveform to a .wav file from a PyTorch tensor or Numpy array.
    Ensures values are clipped within int16 range.
    """
    # 1. Convert to numpy
    if isinstance(data, torch.Tensor):
        audio = data.detach().cpu().numpy()
    else:
        audio = np.array(data)

    # Flatten if necessary
    if audio.ndim > 1:
        audio = audio.flatten()

    # 2. Normalize to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # 3. Clip to avoid overflow
    audio = np.clip(audio, -1.0, 1.0)

    # 4. Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    # 5. Save
    wavfile.write(path, sample_rate, audio_int16)
    print(f"Saved: {path}")