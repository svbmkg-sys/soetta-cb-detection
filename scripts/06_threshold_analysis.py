import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from _config import METRIC_DIR, FIGURE_DIR, REPORT_DIR

warnings.filterwarnings("ignore")


MODEL_FILES = {
    "DNN": "dnn_test_predictions.csv",
    "1D-CNN": "cnn1d_test_predictions.csv",
    "TCN": "tcn_test_predictions.csv",
}

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)


def calculate_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    pod = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "pod_recall": pod,
        "far": far,
        "f1_score": f1,
        "specificity": specificity,
        "csi": csi,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_recommended_threshold(result_df):
    """
    Rekomendasi threshold dibuat dalam beberapa versi:
    1. best_f1: threshold dengan F1-score tertinggi.
    2. best_csi: threshold dengan CSI tertinggi.
    3. operational_balanced: POD minimal 0.75, lalu FAR terkecil.
       Jika tidak ada yang memenuhi POD >= 0.75, pakai threshold dengan F1-score tertinggi.
    """

    best_f1_row = result_df.sort_values(
        ["f1_score", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    best_csi_row = result_df.sort_values(
        ["csi", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    candidates = result_df[result_df["pod_recall"] >= 0.75].copy()

    if len(candidates) > 0:
        operational_row = candidates.sort_values(
            ["far", "f1_score", "accuracy"],
            ascending=[True, False, False],
        ).iloc[0]
        operational_note = "POD >= 0.75, FAR minimum"
    else:
        operational_row = best_f1_row
        operational_note = "Tidak ada threshold dengan POD >= 0.75; menggunakan F1 maksimum"

    return {
        "best_f1_threshold": float(best_f1_row["threshold"]),
        "best_f1_score": float(best_f1_row["f1_score"]),
        "best_csi_threshold": float(best_csi_row["threshold"]),
        "best_csi": float(best_csi_row["csi"]),
        "operational_threshold": float(operational_row["threshold"]),
        "operational_pod": float(operational_row["pod_recall"]),
        "operational_far": float(operational_row["far"]),
        "operational_f1": float(operational_row["f1_score"]),
        "operational_accuracy": float(operational_row["accuracy"]),
        "operational_note": operational_note,
    }


def plot_threshold_metrics(result_df, model_name):
    plt.figure(figsize=(9, 6))

    plt.plot(result_df["threshold"], result_df["accuracy"], label="Accuracy")
    plt.plot(result_df["threshold"], result_df["pod_recall"], label="POD/Recall")
    plt.plot(result_df["threshold"], result_df["far"], label="FAR")
    plt.plot(result_df["threshold"], result_df["f1_score"], label="F1-score")
    plt.plot(result_df["threshold"], result_df["csi"], label="CSI")

    plt.axvline(0.5, linestyle="--", label="Default threshold 0.5")

    plt.title(f"Threshold Analysis - {model_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    output_file = FIGURE_DIR / f"threshold_vs_metrics_{model_name.lower().replace('-', '').replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)


def main():
    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    all_results = []
    recommendations = []

    for model_name, filename in MODEL_FILES.items():
        prediction_file = METRIC_DIR / filename

        if not prediction_file.exists():
            raise FileNotFoundError(
                f"File prediksi tidak ditemukan: {prediction_file}. "
                "Pastikan scripts/05_train_models.py sudah dijalankan."
            )

        pred_df = pd.read_csv(prediction_file)

        y_true = pred_df["y_true"].astype(int).to_numpy()
        y_prob = pred_df["y_prob"].astype(float).to_numpy()

        model_results = []

        for threshold in THRESHOLDS:
            metrics = calculate_metrics_at_threshold(y_true, y_prob, threshold)
            metrics["model"] = model_name
            model_results.append(metrics)

        model_result_df = pd.DataFrame(model_results)

        model_output_file = METRIC_DIR / f"threshold_analysis_{model_name.lower().replace('-', '').replace(' ', '_')}.csv"
        model_result_df.to_csv(model_output_file, index=False)

        print("Saved:", model_output_file)

        plot_threshold_metrics(model_result_df, model_name)

        rec = find_recommended_threshold(model_result_df)
        rec["model"] = model_name
        recommendations.append(rec)

        all_results.append(model_result_df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_file = METRIC_DIR / "threshold_analysis_all_models.csv"
    all_results_xlsx = METRIC_DIR / "threshold_analysis_all_models.xlsx"

    all_results_df.to_csv(all_results_file, index=False)
    all_results_df.to_excel(all_results_xlsx, index=False)

    rec_df = pd.DataFrame(recommendations)

    rec_cols = [
        "model",
        "best_f1_threshold",
        "best_f1_score",
        "best_csi_threshold",
        "best_csi",
        "operational_threshold",
        "operational_pod",
        "operational_far",
        "operational_f1",
        "operational_accuracy",
        "operational_note",
    ]

    rec_df = rec_df[rec_cols]

    rec_file = METRIC_DIR / "threshold_recommendations.csv"
    rec_xlsx = METRIC_DIR / "threshold_recommendations.xlsx"

    rec_df.to_csv(rec_file, index=False)
    rec_df.to_excel(rec_xlsx, index=False)

    report_file = REPORT_DIR / "threshold_analysis_summary.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("THRESHOLD ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Threshold diuji dari 0.10 sampai 0.90 dengan interval 0.01.\n")
        f.write("Evaluasi dilakukan menggunakan data uji tahun 2024.\n\n")
        f.write("Rekomendasi threshold:\n")
        f.write(rec_df.to_string(index=False))
        f.write("\n\nCatatan:\n")
        f.write(
            "- best_f1_threshold adalah threshold dengan F1-score tertinggi.\n"
            "- best_csi_threshold adalah threshold dengan Critical Success Index tertinggi.\n"
            "- operational_threshold dipilih dengan kriteria POD >= 0.75 dan FAR minimum.\n"
            "- Jika tidak ada threshold yang memenuhi POD >= 0.75, digunakan threshold dengan F1-score tertinggi.\n"
        )

    print("\nSaved:")
    print(all_results_file)
    print(all_results_xlsx)
    print(rec_file)
    print(rec_xlsx)
    print(report_file)

    print("\nThreshold recommendations:")
    print(rec_df)


if __name__ == "__main__":
    main()