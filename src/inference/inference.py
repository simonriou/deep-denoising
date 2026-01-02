import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from models.ConvRNNTemporalDenoiser import ConvRNNTemporalDenoiser
from training.dataset import SpeechNoiseDatasetTemporal
from utils.constants import *
from utils.save_wav import save_wav
from utils.compute_snr import compute_snr

# --- 1. Prepare dataset ---
dataset = SpeechNoiseDatasetTemporal(
    clean_dir=CLEAN_TEST_DIR,
    noise_dir=NOISE_TEST_DIR,
    snr_db=TARGET_SNR,
    seq_len=SEQ_LEN,  # must match training
    mode="test"
)

# --- 2. DataLoader ---
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

# --- 3. Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvRNNTemporalDenoiser().to(device)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth"), map_location=device))
model.eval()

# --- 4. Output directory ---
denoised_dir = NOISE_ENHANCED_DIR
os.makedirs(denoised_dir, exist_ok=True)

# --- 5. Inference loop ---
with torch.no_grad():
    for idx, batch in enumerate(loader):
        mixture = batch["mixture"].to(device).unsqueeze(-1)   # [B, T, 1]
        clean_audio = batch["clean"].numpy()[0]

        # Predict clean waveform
        pred_clean = model(mixture)                           # [B, T, 1]
        enhanced_audio = pred_clean[0, :, 0].cpu().numpy()    # [T]

        # Clip to valid range
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        # --- Save audio ---
        if SAVE_DENOISED:
            save_wav(enhanced_audio, os.path.join(denoised_dir, f"denoised_{idx}.wav"), sample_rate=SAMPLE_RATE)

        # --- Compute SNR ---
        min_len = min(len(clean_audio), len(enhanced_audio))
        clean_audio_trim = clean_audio[:min_len]
        enhanced_audio_trim = enhanced_audio[:min_len]

        snr_enhanced = compute_snr(clean_audio_trim, enhanced_audio_trim)
        print(f"File {idx}: Enhanced SNR = {snr_enhanced:.2f} dB")

        # --- Logging ---
        log_csv_dir = os.path.join(LOG_DIR, MODEL_NAME)
        os.makedirs(log_csv_dir, exist_ok=True)
        log_csv_path = os.path.join(log_csv_dir, "inference_snr_log.csv")

        if idx == 0:
            with open(log_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_index", "enhanced_snr_db"])

        with open(log_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, f"{snr_enhanced:.2f}"])

print("Inference Complete.")