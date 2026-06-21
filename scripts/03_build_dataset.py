import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from _config import (
    SYNOP_FILE,
    SOUNDING_FILE,
    PROCESSED_DIR,
    LABEL_HORIZONS,
    NON_FEATURE_COLUMNS,
)

warnings.filterwarnings("ignore")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df


def read_first_valid_sheet(excel_file: Path, required_columns=None) -> pd.DataFrame:
    """
    Membaca sheet pertama yang sesuai.
    Jika required_columns diberikan, script akan mencari sheet yang mengandung kolom tersebut.
    """
    xls = pd.ExcelFile(excel_file)

    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        df = normalize_columns(df)

        if required_columns is None:
            print(f"Using sheet '{sheet}' from {excel_file.name}")
            return df

        available = set(df.columns)
        if set(required_columns).issubset(available):
            print(f"Using sheet '{sheet}' from {excel_file.name}")
            return df

    raise ValueError(
        f"Tidak ditemukan sheet dengan kolom wajib {required_columns} di file {excel_file}"
    )


def load_sounding() -> pd.DataFrame:
    df = read_first_valid_sheet(SOUNDING_FILE, required_columns=["Tanggal", "Jam"])
    df = normalize_columns(df)

    # Tanggal Excel/date -> datetime
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")

    # Jam berupa teks "00" dan "12"
    df["Jam"] = df["Jam"].astype(str).str.strip().str.zfill(2)

    # Buat waktu sounding UTC
    df["sounding_time_utc"] = pd.to_datetime(
        df["Tanggal"].dt.strftime("%Y-%m-%d") + " " + df["Jam"] + ":00:00",
        errors="coerce",
    )

    df = df.dropna(subset=["sounding_time_utc"]).copy()

    df["year"] = df["sounding_time_utc"].dt.year
    df["month"] = df["sounding_time_utc"].dt.month
    df["day"] = df["sounding_time_utc"].dt.day
    df["hour"] = df["sounding_time_utc"].dt.hour

    # Buang duplikasi waktu sounding jika ada
    before = len(df)
    df = df.sort_values("sounding_time_utc")
    df = df.drop_duplicates(subset=["sounding_time_utc"], keep="last")
    after = len(df)

    print(f"Sounding rows: {before} -> {after} after removing duplicate times")
    print("Sounding period:", df["sounding_time_utc"].min(), "to", df["sounding_time_utc"].max())

    return df


def parse_cumulonimbus_value(x) -> bool:
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)

    s = str(x).strip().lower()

    true_values = {"true", "1", "yes", "y", "ya", "cb", "cumulonimbus"}
    false_values = {"false", "0", "no", "n", "tidak", "", "nan", "none"}

    if s in true_values:
        return True
    if s in false_values:
        return False

    # Fallback: jika teks mengandung cb/cumulonimbus
    if "cumulonimbus" in s or s == "cb":
        return True

    return False


def load_synop() -> pd.DataFrame:
    df = read_first_valid_sheet(SYNOP_FILE, required_columns=["TIMESTAMP_UTC"])
    df = normalize_columns(df)

    df["TIMESTAMP_UTC"] = pd.to_datetime(df["TIMESTAMP_UTC"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP_UTC"]).copy()

    if "cumulonimbus" in df.columns:
        df["cb_observed"] = df["cumulonimbus"].apply(parse_cumulonimbus_value)
    elif "CLOUD LOW TYPE CL" in df.columns:
        # Fallback jika kolom cumulonimbus tidak ada.
        # Catatan: pada kode awan rendah SYNOP, CL=9 berkaitan dengan Cb.
        df["cb_observed"] = df["CLOUD LOW TYPE CL"].astype(str).str.strip().eq("9")
    else:
        raise ValueError(
            "Tidak ditemukan kolom 'cumulonimbus' atau 'CLOUD LOW TYPE CL' pada data SYNOP."
        )

    df = df.sort_values("TIMESTAMP_UTC")
    df = df.drop_duplicates(subset=["TIMESTAMP_UTC"], keep="last")

    print("SYNOP rows:", len(df))
    print("SYNOP period:", df["TIMESTAMP_UTC"].min(), "to", df["TIMESTAMP_UTC"].max())
    print("CB observed count:")
    print(df["cb_observed"].value_counts(dropna=False))

    return df


def assign_labels(sounding: pd.DataFrame, synop: pd.DataFrame) -> pd.DataFrame:
    """
    Label:
    cb_h3  = 1 jika ada CB pada interval [t0, t0 + 3 jam)
    cb_h6  = 1 jika ada CB pada interval [t0, t0 + 6 jam)
    cb_h12 = 1 jika ada CB pada interval [t0, t0 + 12 jam)
    """
    sounding = sounding.copy()
    cb_times = synop.loc[synop["cb_observed"], "TIMESTAMP_UTC"].sort_values().to_numpy()

    if len(cb_times) == 0:
        raise ValueError("Tidak ada kejadian CB pada data SYNOP. Cek kembali kolom label.")

    sounding_times = sounding["sounding_time_utc"].to_numpy()

    for horizon in LABEL_HORIZONS:
        labels = []

        for t0 in sounding_times:
            t1 = pd.Timestamp(t0) + pd.Timedelta(hours=horizon)

            left = np.searchsorted(cb_times, np.datetime64(t0), side="left")
            right = np.searchsorted(cb_times, np.datetime64(t1), side="left")

            labels.append(int(right > left))

        col = f"cb_h{horizon}"
        sounding[col] = labels

        print(f"\nLabel distribution for {col}:")
        print(sounding[col].value_counts(dropna=False))

    return sounding


def clean_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Membersihkan fitur numerik:
    - konversi kolom kandidat ke numerik jika memungkinkan
    - -9999 dan nilai ekstrem tertentu dianggap missing
    - buang kolom fitur yang missing-nya terlalu banyak
    """
    df = df.copy()

    candidate_cols = [col for col in df.columns if col not in NON_FEATURE_COLUMNS]

    numeric_features = []

    for col in candidate_cols:
        converted = pd.to_numeric(df[col], errors="coerce")

        # Kolom dianggap fitur numerik jika minimal 50% datanya bisa dibaca numerik
        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.50:
            df[col] = converted
            numeric_features.append(col)

    # Ubah kode missing umum menjadi NaN
    df[numeric_features] = df[numeric_features].replace(
        [-9999, -9999.0, 9999, 9999.0, np.inf, -np.inf],
        np.nan,
    )

    # Buang kolom dengan missing > 40%
    missing_ratio = df[numeric_features].isna().mean()
    selected_features = missing_ratio[missing_ratio <= 0.40].index.tolist()

    removed_features = sorted(set(numeric_features) - set(selected_features))

    print("\nTotal numeric candidate features:", len(numeric_features))
    print("Selected features:", len(selected_features))
    print("Removed features due to missing > 40%:", removed_features)

    return df, selected_features


def main():
    print("=" * 80)
    print("BUILDING CB DETECTION DATASET")
    print("=" * 80)

    sounding = load_sounding()
    synop = load_synop()

    dataset = assign_labels(sounding, synop)
    dataset, feature_columns = clean_numeric_features(dataset)

    dataset = dataset.sort_values("sounding_time_utc").reset_index(drop=True)

    output_csv = PROCESSED_DIR / "cb_dataset_tabular.csv"
    output_xlsx = PROCESSED_DIR / "cb_dataset_tabular.xlsx"
    metadata_json = PROCESSED_DIR / "dataset_metadata.json"

    dataset.to_csv(output_csv, index=False)
    dataset.to_excel(output_xlsx, index=False)

    metadata = {
        "sounding_file": str(SOUNDING_FILE.name),
        "synop_file": str(SYNOP_FILE.name),
        "n_rows": int(len(dataset)),
        "period_start": str(dataset["sounding_time_utc"].min()),
        "period_end": str(dataset["sounding_time_utc"].max()),
        "label_columns": [f"cb_h{h}" for h in LABEL_HORIZONS],
        "main_label": "cb_h12",
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
    }

    with open(metadata_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("\nSaved:")
    print(output_csv)
    print(output_xlsx)
    print(metadata_json)

    print("\nFinal dataset preview:")
    print(dataset.head())

    print("\nFeature columns:")
    for col in feature_columns:
        print("-", col)


if __name__ == "__main__":
    main()