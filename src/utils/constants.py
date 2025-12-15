from pathlib import Path

"""
These are the constants used throughout the project.
"""

SAMPLE_RATE = 16000
TARGET_SNR = 0.0 # dB
MIN_DB_CLIP = 80.0

N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512

ROOT = Path(__file__).parent.parent.parent.resolve() # root folder

DATA_DIR = ROOT / "data"

CLEAN_DIR = DATA_DIR / "speech"
NOISE_DIR = DATA_DIR / "noise"
MODEL_DIR = DATA_DIR / "models"

LOG_DIR = ROOT / "experiments" / "logs"
CHECKPOINT_DIR = ROOT / "experiments" / "checkpoints"

EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001

DEBUG = False