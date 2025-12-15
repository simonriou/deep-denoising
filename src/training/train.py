import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.constants import *
from training.dataset import SpeechNoiseDataset
from models.DenoiseUNet import DenoiseUNet
from utils.pad_collate import pad_collate

"""
This is the main training script for the speech denoising model.
It sets up the dataset, dataloader, model, loss function, and optimizer.
It runs the training loop for a specified number of epochs, printing progress and saving model checkpoints.

The criterion used is Binary Cross-Entropy Loss (BCELoss) since the model outputs ideal binary masks (IBM).
The Adam optimizer is used for training.
"""

def train(session_name: str):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(NOISE_DIR, exist_ok=True)
    if not glob.glob(f"{CLEAN_DIR}/*.pt"):
        print("Error: No clean data found. Please add .pt files to the clean data directory.")
        return

    # 2. Load Data
    dataset = SpeechNoiseDataset(CLEAN_DIR, NOISE_DIR, snr_db=TARGET_SNR)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=pad_collate,
        pin_memory=(device.type == 'cuda') 
    )
    
    # 3. Model & Loss
    model = DenoiseUNet().to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create checkpoint directory for this session
    checkpoints_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, f"{session_name}_training_log.csv")

    # Write CSV header
    with open(log_file_path, mode='w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss"])

    print("Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS): 
        epoch_loss = 0
        for batch_idx, (features, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            features = features.to(device)
            targets = targets.to(device)

            outputs = model(features)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"denoise_cnn_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Log to CSV
        with open(log_file_path, mode='a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])

        # If final epoch, also save final model
        if epoch == EPOCHS - 1:
            final_model_path = os.path.join(MODEL_DIR, f"{session_name}_final.pth")
            torch.save(model.state_dict(), final_model_path)

    print("Training Complete.")


if __name__ == "__main__":
    # Ask for session name
    session_name = input("Enter a session name for this training run: ").strip()
    if not session_name:
        print("Session name cannot be empty. Exiting.")
    elif os.path.exists(os.path.join(CHECKPOINT_DIR, session_name)):
        overwrite = input(f"Session '{session_name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite == 'y':
            train(session_name)
        else:
            print("Exiting without training.")
    else:
        train(session_name)