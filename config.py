from pathlib import Path

SEED = 42
DATA_MODE = "nbaiot"

DATA_ROOT = "data/raw/N-BaIoT"

DEVICE_NAME = "Philips_B120N10_Baby_Monitor"

MAX_BENIGN_ROWS = 25000
MAX_MALWARE_ROWS = 25000

TEST_SIZE = 0.20
VAL_SIZE = 0.20

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

REPRESENTATIVE_SAMPLES = 300
TIMED_SAMPLES = 200
WARMUP_SAMPLES = 20
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
