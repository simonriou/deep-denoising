import glob
import torch
from torch.utils.data import DataLoader
from models.DenoiseUNet import DenoiseUNet
from training.dataset import SpeechNoiseDataset
from utils.constants import *
from utils.save_wav import save_wav
import os
import numpy as np

# 1. Prepare test clean files and noise files
test_clean_files = glob.glob(str(CLEAN_TEST_DIR / '*.pt'))
noise_files = glob.glob(str(NOISE_TEST_DIR / '*.pt'))

# 2. Create Dataset instance
dataset = SpeechNoiseDataset(clean_dir=CLEAN_TEST_DIR, noise_dir=NOISE_TEST_DIR, snr_db=TARGET_SNR, mode='test')

# 3. DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 4. Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoiseUNet().to(device)
model.load_state_dict(torch.load(MODEL_DIR/'unet_1_final.pth', map_location=device))
model.eval()

# 5. Output directory for denoised audio
denoised_dir = NOISE_ENHANCED_DIR
os.makedirs(denoised_dir, exist_ok=True)

# 6. Inference loop
with torch.no_grad():
    for idx, (features, ibm, mixture_mag, mixture_phase) in enumerate(loader):
        features = features.to(device)
        pred_mask = model(features)  # [1, Freq, Time]

        # Apply predicted mask
        pred_mask = model(features)  # [1, 1, Freq, Time]
        enhanced_mag = mixture_mag.to(device) * pred_mask  # keep shape [1, 1, Freq, Time]

        # Remove channel dimension for istft: istft expects [batch, freq, time] or [freq, time]
        # So pick first batch and first channel
        complex_enhanced = enhanced_mag[0,0] * torch.exp(1j * mixture_phase[0,0])  # [Freq, Time]

        window = torch.hann_window(WIN_LENGTH)
        enhanced_audio = torch.istft(complex_enhanced, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH,
                                    win_length=WIN_LENGTH,
                                    window=window)

        enhanced_audio = enhanced_audio.cpu().numpy()
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        # Save
        save_wav(enhanced_audio, denoised_dir / f"denoised_{idx}.wav", sample_rate=SAMPLE_RATE)