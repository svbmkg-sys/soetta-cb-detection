import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from _config import METRIC_DIR, FIGURE_DIR, REPORT_DIR

warnings.filterwarnings("ignore")


MODEL_FILES = {
    "DNN": {
        "model_key": "dnn",
        "val": "dnn_val_predictions.csv",
        "test": "dnn_test_predictions.csv",
    },
    "1D-CNN": {
        "model_key": "cnn1d",
        "val": "cnn1d_val_predictions.csv",
        "test": "cnn1d_test_predictions.csv",
    },
    "TCN": {
        "model_key": "tcn",
        "val": "tcn_val_predictions.csv",
        "test": "tcn_test_predictions.csv",
    },
}

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)


def calculate_metrics(y_true, y_prob, threshold):
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

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "pod_recall": float(pod),
        "far": float(far),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "csi": float(csi),
        "auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_all_thresholds(y_true, y_prob, model_name, split_name):
    rows = []

    for threshold in THRESHOLDS:
        row = calculate_metrics(y_true, y_prob, threshold)
        row["model"] = model_name
        row["split"] = split_name
        rows.append(row)

    return pd.DataFrame(rows)


def choose_threshold_from_validation(val_result_df):
    """
    Tiga pilihan threshold:
    1. best_f1       : F1-score maksimum pada validation.
    2. best_csi      : CSI maksimum pada validation.
    3. operational   : POD validation >= 0.75, lalu FAR minimum.
                      Jika tidak ada, gunakan best_f1.
    """

    best_f1 = val_result_df.sort_values(
        ["f1_score", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    best_csi = val_result_df.sort_values(
        ["csi", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    candidates = val_result_df[val_result_df["pod_recall"] >= 0.75].copy()

    if len(candidates) > 0:
        operational = candidates.sort_values(
            ["far", "f1_score", "accuracy"],
            ascending=[True, False, False],
        ).iloc[0]
        operational_note = "Dipilih dari validation: POD >= 0.75, FAR minimum"
    else:
        operational = best_f1
        operational_note = "Tidak ada threshold validation dengan POD >= 0.75; menggunakan F1 maksimum"

    return {
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1_validation_f1": float(best_f1["f1_score"]),
        "best_csi_threshold": float(best_csi["threshold"]),
        "best_csi_validation_csi": float(best_csi["csi"]),
        "operational_threshold": float(operational["threshold"]),
        "operational_validation_pod": float(operational["pod_recall"]),
        "operational_validation_far": float(operational["far"]),
        "operational_validation_f1": float(operational["f1_score"]),
        "operational_validation_accuracy": float(operational["accuracy"]),
        "operational_note": operational_note,
    }


def plot_validation_threshold_curve(val_result_df, model_name):
    plt.figure(figsize=(9, 6))

    plt.plot(val_result_df["threshold"], val_result_df["accuracy"], label="Accuracy")
    plt.plot(val_result_df["threshold"], val_result_df["pod_recall"], label="POD/Recall")
    plt.plot(val_result_df["threshold"], val_result_df["far"], label="FAR")
    plt.plot(val_result_df["threshold"], val_result_df["f1_score"], label="F1-score")
    plt.plot(val_result_df["threshold"], val_result_df["csi"], label="CSI")

    plt.axvline(0.5, linestyle="--", label="Default threshold 0.5")

    plt.title(f"Validation Threshold Analysis - {model_name}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    safe_name = model_name.lower().replace("-", "").replace(" ", "_")
    output_file = FIGURE_DIR / f"validation_threshold_vs_metrics_{safe_name}.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)


def main():
    print("=" * 80)
    print("THRESHOLD SELECTION FROM VALIDATION AND FINAL TEST EVALUATION")
    print("=" * 80)

    all_validation_thresholds = []
    threshold_recommendations = []
    final_test_rows = []

    for model_name, file_info in MODEL_FILES.items():
        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        val_file = METRIC_DIR / file_info["val"]
        test_file = METRIC_DIR / file_info["test"]

        if not val_file.exists():
            raise FileNotFoundError(
                f"File validation prediction tidak ditemukan: {val_file}. "
                "Jalankan ulang scripts/05_train_models.py yang sudah diperbarui."
            )

        if not test_file.exists():
            raise FileNotFoundError(
                f"File test prediction tidak ditemukan: {test_file}. "
                "Jalankan scripts/05_train_models.py terlebih dahulu."
            )

        val_pred = pd.read_csv(val_file)
        test_pred = pd.read_csv(test_file)

        y_val = val_pred["y_true"].astype(int).to_numpy()
        p_val = val_pred["y_prob"].astype(float).to_numpy()

        y_test = test_pred["y_true"].astype(int).to_numpy()
        p_test = test_pred["y_prob"].astype(float).to_numpy()

        # 1. Cari threshold pada validation
        val_result_df = evaluate_all_thresholds(
            y_true=y_val,
            y_prob=p_val,
            model_name=model_name,
            split_name="validation",
        )

        all_validation_thresholds.append(val_result_df)

        safe_name = model_name.lower().replace("-", "").replace(" ", "_")
        val_threshold_file = METRIC_DIR / f"validation_threshold_analysis_{safe_name}.csv"
        val_result_df.to_csv(val_threshold_file, index=False)
        print("Saved:", val_threshold_file)

        plot_validation_threshold_curve(val_result_df, model_name)

        rec = choose_threshold_from_validation(val_result_df)
        rec["model"] = model_name
        threshold_recommendations.append(rec)

        # 2. Evaluasi test dengan tiga threshold dari validation
        threshold_options = {
            "default_0_50": 0.50,
            "best_f1_from_validation": rec["best_f1_threshold"],
            "best_csi_from_validation": rec["best_csi_threshold"],
            "operational_from_validation": rec["operational_threshold"],
        }

        for threshold_name, threshold_value in threshold_options.items():
            test_metrics = calculate_metrics(
                y_true=y_test,
                y_prob=p_test,
                threshold=threshold_value,
            )

            test_metrics["model"] = model_name
            test_metrics["threshold_source"] = threshold_name
            test_metrics["threshold"] = float(threshold_value)

            final_test_rows.append(test_metrics)

    # Gabungan seluruh threshold validation
    all_val_df = pd.concat(all_validation_thresholds, ignore_index=True)

    all_val_csv = METRIC_DIR / "validation_threshold_analysis_all_models.csv"
    all_val_xlsx = METRIC_DIR / "validation_threshold_analysis_all_models.xlsx"

    all_val_df.to_csv(all_val_csv, index=False)
    all_val_df.to_excel(all_val_xlsx, index=False)

    # Rekomendasi threshold dari validation
    rec_df = pd.DataFrame(threshold_recommendations)

    rec_cols = [
        "model",
        "best_f1_threshold",
        "best_f1_validation_f1",
        "best_csi_threshold",
        "best_csi_validation_csi",
        "operational_threshold",
        "operational_validation_pod",
        "operational_validation_far",
        "operational_validation_f1",
        "operational_validation_accuracy",
        "operational_note",
    ]
    rec_df = rec_df[rec_cols]

    rec_csv = METRIC_DIR / "threshold_recommendations_from_validation.csv"
    rec_xlsx = METRIC_DIR / "threshold_recommendations_from_validation.xlsx"

    rec_df.to_csv(rec_csv, index=False)
    rec_df.to_excel(rec_xlsx, index=False)

    # Final test metrics
    final_test_df = pd.DataFrame(final_test_rows)

    final_cols = [
        "model",
        "threshold_source",
        "threshold",
        "accuracy",
        "precision",
        "pod_recall",
        "far",
        "f1_score",
        "specificity",
        "csi",
        "auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    final_test_df = final_test_df[final_cols]

    final_csv = METRIC_DIR / "final_test_metrics_with_validation_threshold.csv"
    final_xlsx = METRIC_DIR / "final_test_metrics_with_validation_threshold.xlsx"

    final_test_df.to_csv(final_csv, index=False)
    final_test_df.to_excel(final_xlsx, index=False)

    # Report text
    report_file = REPORT_DIR / "threshold_from_validation_summary.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("THRESHOLD SELECTION FROM VALIDATION SET\n")
        f.write("=" * 80 + "\n\n")
        f.write("Threshold dipilih menggunakan data validasi tahun 2023.\n")
        f.write("Threshold terpilih kemudian diterapkan pada data uji tahun 2024.\n")
        f.write("Dengan cara ini, data uji tidak digunakan untuk memilih threshold.\n\n")

        f.write("REKOMENDASI THRESHOLD DARI VALIDATION\n")
        f.write("-" * 80 + "\n")
        f.write(rec_df.to_string(index=False))
        f.write("\n\n")

        f.write("HASIL FINAL PADA TEST SET 2024\n")
        f.write("-" * 80 + "\n")
        f.write(final_test_df.to_string(index=False))
        f.write("\n\n")

        f.write("Catatan:\n")
        f.write("- default_0_50 adalah threshold standar 0.50.\n")
        f.write("- best_f1_from_validation menggunakan threshold dengan F1-score tertinggi pada validation.\n")
        f.write("- best_csi_from_validation menggunakan threshold dengan CSI tertinggi pada validation.\n")
        f.write("- operational_from_validation memilih threshold dengan POD validation >= 0.75 dan FAR minimum.\n")
        f.write("- Jika tidak ada threshold yang memenuhi POD >= 0.75, digunakan threshold dengan F1-score tertinggi.\n")

    print("\nSaved:")
    print(all_val_csv)
    print(all_val_xlsx)
    print(rec_csv)
    print(rec_xlsx)
    print(final_csv)
    print(final_xlsx)
    print(report_file)

    print("\nThreshold recommendations from validation:")
    print(rec_df)

    print("\nFinal test metrics:")
    print(final_test_df)


if __name__ == "__main__":
    main()