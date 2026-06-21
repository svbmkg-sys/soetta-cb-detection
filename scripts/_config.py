from pathlib import Path

# =========================
# PATH CONFIGURATION
# =========================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_DIR / "data" / "raw"
INTERIM_DIR = PROJECT_DIR / "data" / "interim"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

SYNOP_FILE = RAW_DIR / "sinop_01.xlsx"
SOUNDING_FILE = RAW_DIR / "sounding_indices_96749_full.xlsx"

for folder in [
    RAW_DIR,
    INTERIM_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    MODEL_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================
# RESEARCH CONFIGURATION
# =========================

MAIN_TARGET = "cb_h12"

LABEL_HORIZONS = [3, 6, 12]

TRAIN_YEARS = [2020, 2021, 2022]
VAL_YEARS = [2023]
TEST_YEARS = [2024]

RANDOM_SEED = 42

# Untuk 1D-CNN dan TCN.
# window=3 berarti model melihat 3 data sounding terakhir.
SEQUENCE_WINDOW = 3

# Kolom non-fitur yang tidak boleh masuk model
NON_FEATURE_COLUMNS = {
    "Tanggal",
    "Jam",
    "sounding_time_utc",
    "source_file",
    "year",
    "month",
    "day",
    "hour",
    "cb_h3",
    "cb_h6",
    "cb_h12",
    "Station number",
    "Station latitude",
    "Station longitude",
    "Station elevation",
}