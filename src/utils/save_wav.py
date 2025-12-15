import torch
import scipy.io.wavfile as wavfile
import numpy as np

"""
This function saves a waveform to a .wav file from a PyTorch tensor or Numpy array.
It is used for debugging and verification purposes.
"""

def save_wav(data, path, sample_rate=16000):
    """
    Saves a waveform to a .wav file from a PyTorch tensor or Numpy array.
    """
    # 1. Check if input is a PyTorch Tensor
    if isinstance(data, torch.Tensor):
        # Move to CPU and convert to Numpy
        audio = data.detach().cpu().numpy()
    else:
        # It's already a Numpy array
        audio = data

    # 2. Normalize and convert to int16
    # Ensure we don't divide by zero if silence
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    
    # Scale to int16 range
    audio = (audio * 32767).astype(np.int16)
    
    # 3. Save
    wavfile.write(path, sample_rate, audio)
    print(f"Saved: {path}")