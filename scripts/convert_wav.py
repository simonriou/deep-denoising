import os
import glob
import torch
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from tqdm import tqdm

from src.utils.constants import *

def convert_wavs():
    os.makedirs(CLEAN_TEST_DIR, exist_ok=True)
    wav_files = glob.glob(os.path.join(RAW_TEST_DIR, '*.wav'))
    
    if not wav_files:
        print(f"No .wav files found in {RAW_TEST_DIR}")
        return

    print(f"Found {len(wav_files)} files. Converting to {SAMPLE_RATE}Hz int16 .pt...")

    for wav_path in tqdm(wav_files):
        try:
            # 1. Read WAV file
            # sr: sample rate, data: numpy array
            sr, data = wav.read(wav_path)
            
            # 2. Convert to Mono if Stereo
            # If shape is (samples, channels), average to (samples,)
            if data.ndim > 1:
                data = data.mean(axis=1)
            
            # 3. Resample if necessary
            if sr != SAMPLE_RATE:
                # Calculate new length
                num_samples = int(len(data) * SAMPLE_RATE / sr)
                # Resample (returns float data)
                data = signal.resample(data, num_samples)
            
            # 4. Ensure Data is Float for Scaling/Normalization logic
            # (scipy reads int16 as int, float32 as float)
            data = data.astype(np.float32)

            # 5. Normalize and Convert to Int16
            # Check if data is essentially in float -1..1 range or huge int range
            max_val = np.abs(data).max()
            
            if max_val > 0:
                if max_val <= 1.0:
                    # It was likely float32 (-1 to 1). Scale to int16 range.
                    data = data * 32767
                else:
                    # It was likely already integer-like (e.g. read from int16 wav).
                    # If we resampled, it might have drifted slightly beyond 32767, so we clip.
                    data = np.clip(data, -32768, 32767)
            
            # Final cast to int16
            data_int16 = data.astype(np.int16)

            # 6. Save as Torch Tensor
            filename = os.path.basename(wav_path).replace('.wav', '.pt')
            out_path = os.path.join(CLEAN_TEST_DIR, filename)
            
            # Create tensor (copy to ensure no negative strides issues from resampling)
            tensor = torch.from_numpy(data_int16.copy())
            torch.save(tensor, out_path)
            
            print(f"Processed: {filename} | Shape: {tensor.shape}")

        except Exception as e:
            print(f"Failed to process {wav_path}: {e}")

if __name__ == "__main__":
    # Create dummy folder for testing
    if not os.path.exists(RAW_TEST_DIR):
        os.makedirs(RAW_TEST_DIR)
        print(f"Created {RAW_TEST_DIR}. Please put your .wav files there.")
        
    convert_wavs()