import glob
import torch
from torch.utils.data import DataLoader
from models.DenoiseUNet import DenoiseUNet
from training.dataset import SpeechNoiseDataset
from utils.constants import *
from utils.save_wav import save_wav
from utils.compute_snr import compute_snr
import os
import csv
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
model.load_state_dict(torch.load(MODEL_DIR/f'{MODEL_NAME}.pth', map_location=device))
model.eval()

# 5. Output directory for denoised audio
denoised_dir = NOISE_ENHANCED_DIR
os.makedirs(denoised_dir, exist_ok=True)

# 6. Inference loop
with torch.no_grad():
    for idx, (features, ibm, mixture_mag, mixture_phase, clean_audio) in enumerate(loader):
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

        complex_initial = mixture_mag[0,0] * torch.exp(1j * mixture_phase[0,0])
        initial_audio = torch.istft(complex_initial, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH,
                                    win_length=WIN_LENGTH,
                                    window=window)
        
        initial_audio = initial_audio.cpu().numpy()
        initial_audio = np.clip(initial_audio, -1.0, 1.0)

        # Save
        if SAVE_DENOISED:
            save_wav(enhanced_audio, denoised_dir / f"denoised_{idx}.wav", sample_rate=SAMPLE_RATE)
        if SAVE_NOISY:
            save_wav(initial_audio, denoised_dir / f"noisy_{idx}.wav", sample_rate=SAMPLE_RATE)

        # Compute metrics (initial vs. enhanced SNR) and log them

        log_csv_dir = os.path.join(LOG_DIR, MODEL_NAME)
        os.makedirs(log_csv_dir, exist_ok=True)
        log_csv_path = os.path.join(log_csv_dir, "inference_snr_log.csv")

        # Write CSV Header
        if idx == 0:
            with open(log_csv_path, mode='w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_index", "noisy_snr_db", "enhanced_snr_db"])

        # Match sizes for SNR computation
        min_len = min(len(clean_audio), len(initial_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        initial_audio = initial_audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]

        snr_noisy = compute_snr(clean_audio.numpy(), initial_audio)
        snr_enhanced = compute_snr(clean_audio.numpy(), enhanced_audio)

        print(f"File {idx}: Noisy SNR: {snr_noisy:.2f} dB, Enhanced SNR: {snr_enhanced:.2f} dB")

        # Log to CSV
        with open(log_csv_path, mode='a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, f"{snr_noisy:.2f}", f"{snr_enhanced:.2f}"])

print("Inference Complete.")