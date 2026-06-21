import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _config import PROCESSED_DIR, FIGURE_DIR, REPORT_DIR, MAIN_TARGET

warnings.filterwarnings("ignore")


DATASET_FILE = PROCESSED_DIR / "cb_dataset_tabular.csv"
METADATA_FILE = PROCESSED_DIR / "dataset_metadata.json"


def save_bar_chart(series, title, xlabel, ylabel, output_file):
    plt.figure(figsize=(8, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def main():
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(DATASET_FILE, parse_dates=["sounding_time_utc"])

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]

    # =========================
    # BASIC SUMMARY
    # =========================

    summary = {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "period_start": str(df["sounding_time_utc"].min()),
        "period_end": str(df["sounding_time_utc"].max()),
        "target": MAIN_TARGET,
        "target_distribution": df[MAIN_TARGET].value_counts(dropna=False).to_dict(),
    }

    summary_file = REPORT_DIR / "eda_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("Saved:", summary_file)

    # =========================
    # MISSING VALUE REPORT
    # =========================

    missing = (
        df[feature_cols]
        .isna()
        .sum()
        .to_frame("missing_count")
    )
    missing["missing_percent"] = missing["missing_count"] / len(df) * 100
    missing = missing.sort_values("missing_percent", ascending=False)

    missing_file = REPORT_DIR / "missing_value_report.csv"
    missing.to_csv(missing_file)
    print("Saved:", missing_file)

    # =========================
    # DESCRIPTIVE STATISTICS
    # =========================

    desc = df[feature_cols].describe().T
    desc_file = REPORT_DIR / "feature_descriptive_statistics.csv"
    desc.to_csv(desc_file)
    print("Saved:", desc_file)

    # =========================
    # LABEL DISTRIBUTION
    # =========================

    label_counts = df[MAIN_TARGET].value_counts().sort_index()
    save_bar_chart(
        label_counts,
        title=f"Distribusi Label {MAIN_TARGET}",
        xlabel="Label CB",
        ylabel="Jumlah Data",
        output_file=FIGURE_DIR / "label_distribution_cb_h12.png",
    )

    # =========================
    # YEARLY LABEL DISTRIBUTION
    # =========================

    yearly = df.groupby(["year", MAIN_TARGET]).size().unstack(fill_value=0)
    yearly_file = REPORT_DIR / "yearly_label_distribution.csv"
    yearly.to_csv(yearly_file)
    print("Saved:", yearly_file)

    yearly.plot(kind="bar", figsize=(9, 5))
    plt.title(f"Distribusi Label {MAIN_TARGET} per Tahun")
    plt.xlabel("Tahun")
    plt.ylabel("Jumlah Data")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "yearly_label_distribution.png", dpi=300)
    plt.close()

    # =========================
    # MONTHLY CB FREQUENCY
    # =========================

    monthly = df.groupby(["month", MAIN_TARGET]).size().unstack(fill_value=0)
    monthly_file = REPORT_DIR / "monthly_label_distribution.csv"
    monthly.to_csv(monthly_file)
    print("Saved:", monthly_file)

    monthly.plot(kind="bar", figsize=(10, 5))
    plt.title(f"Distribusi Label {MAIN_TARGET} per Bulan")
    plt.xlabel("Bulan")
    plt.ylabel("Jumlah Data")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monthly_label_distribution.png", dpi=300)
    plt.close()

    # =========================
    # CORRELATION MATRIX
    # =========================

    corr = df[feature_cols].corr(numeric_only=True)

    corr_file = REPORT_DIR / "feature_correlation_matrix.csv"
    corr.to_csv(corr_file)
    print("Saved:", corr_file)

    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=90, fontsize=6)
    plt.yticks(range(len(feature_cols)), feature_cols, fontsize=6)
    plt.title("Matriks Korelasi Fitur Indeks Stabilitas")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_correlation_matrix.png", dpi=300)
    plt.close()

    # =========================
    # FEATURE MEAN BY CLASS
    # =========================

    feature_mean_by_class = df.groupby(MAIN_TARGET)[feature_cols].mean().T
    feature_mean_file = REPORT_DIR / "feature_mean_by_cb_class.csv"
    feature_mean_by_class.to_csv(feature_mean_file)
    print("Saved:", feature_mean_file)

    print("\nEDA completed.")
    print("Figures saved to:", FIGURE_DIR)
    print("Reports saved to:", REPORT_DIR)


if __name__ == "__main__":
    main()