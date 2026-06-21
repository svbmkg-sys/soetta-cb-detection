import warnings

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

from _config import METRIC_DIR, REPORT_DIR

warnings.filterwarnings("ignore")


MODEL_FILES = {
    "1D-CNN_w3": {
        "val": "cnn1d_w3_val_predictions.csv",
        "test": "cnn1d_w3_test_predictions.csv",
    },
    "1D-CNN_w5": {
        "val": "cnn1d_w5_val_predictions.csv",
        "test": "cnn1d_w5_test_predictions.csv",
    },
    "1D-CNN_w7": {
        "val": "cnn1d_w7_val_predictions.csv",
        "test": "cnn1d_w7_test_predictions.csv",
    },
    "TCN_w3": {
        "val": "tcn_w3_val_predictions.csv",
        "test": "tcn_w3_test_predictions.csv",
    },
    "TCN_w5": {
        "val": "tcn_w5_val_predictions.csv",
        "test": "tcn_w5_test_predictions.csv",
    },
    "TCN_w7": {
        "val": "tcn_w7_val_predictions.csv",
        "test": "tcn_w7_test_predictions.csv",
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
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
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


def evaluate_thresholds(y_true, y_prob, model_name, split_name):
    rows = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(y_true, y_prob, threshold)
        metrics["model"] = model_name
        metrics["split"] = split_name
        rows.append(metrics)

    return pd.DataFrame(rows)


def choose_threshold_from_validation(val_df):
    best_f1 = val_df.sort_values(
        ["f1_score", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    best_csi = val_df.sort_values(
        ["csi", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    candidates = val_df[val_df["pod_recall"] >= 0.75].copy()

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


def main():
    print("=" * 80)
    print("SEQUENCE WINDOW THRESHOLD SELECTION FROM VALIDATION")
    print("=" * 80)

    all_validation_rows = []
    recommendation_rows = []
    final_test_rows = []

    for model_name, files in MODEL_FILES.items():
        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        val_file = METRIC_DIR / files["val"]
        test_file = METRIC_DIR / files["test"]

        if not val_file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {val_file}")

        if not test_file.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {test_file}")

        val_pred = pd.read_csv(val_file)
        test_pred = pd.read_csv(test_file)

        y_val = val_pred["y_true"].astype(int).to_numpy()
        p_val = val_pred["y_prob"].astype(float).to_numpy()

        y_test = test_pred["y_true"].astype(int).to_numpy()
        p_test = test_pred["y_prob"].astype(float).to_numpy()

        # Threshold analysis pada validation
        val_threshold_df = evaluate_thresholds(
            y_true=y_val,
            y_prob=p_val,
            model_name=model_name,
            split_name="validation",
        )

        all_validation_rows.append(val_threshold_df)

        safe_name = model_name.lower().replace("-", "").replace(" ", "_")
        val_threshold_file = METRIC_DIR / f"sequence_threshold_validation_{safe_name}.csv"
        val_threshold_df.to_csv(val_threshold_file, index=False)

        rec = choose_threshold_from_validation(val_threshold_df)
        rec["model"] = model_name
        recommendation_rows.append(rec)

        threshold_options = {
            "default_0_50": 0.50,
            "best_f1_from_validation": rec["best_f1_threshold"],
            "best_csi_from_validation": rec["best_csi_threshold"],
            "operational_from_validation": rec["operational_threshold"],
        }

        for threshold_source, threshold_value in threshold_options.items():
            test_metrics = calculate_metrics(
                y_true=y_test,
                y_prob=p_test,
                threshold=threshold_value,
            )

            test_metrics["model"] = model_name
            test_metrics["threshold_source"] = threshold_source

            final_test_rows.append(test_metrics)

    all_validation_df = pd.concat(all_validation_rows, ignore_index=True)
    rec_df = pd.DataFrame(recommendation_rows)
    final_test_df = pd.DataFrame(final_test_rows)

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

    rec_df = rec_df[rec_cols]
    final_test_df = final_test_df[final_cols]

    all_validation_file = METRIC_DIR / "sequence_threshold_validation_all_models.csv"
    rec_file = METRIC_DIR / "sequence_threshold_recommendations_from_validation.csv"
    final_file = METRIC_DIR / "sequence_final_test_metrics_with_validation_threshold.csv"

    all_validation_df.to_csv(all_validation_file, index=False)
    rec_df.to_csv(rec_file, index=False)
    final_test_df.to_csv(final_file, index=False)

    all_validation_df.to_excel(
        METRIC_DIR / "sequence_threshold_validation_all_models.xlsx",
        index=False,
    )
    rec_df.to_excel(
        METRIC_DIR / "sequence_threshold_recommendations_from_validation.xlsx",
        index=False,
    )
    final_test_df.to_excel(
        METRIC_DIR / "sequence_final_test_metrics_with_validation_threshold.xlsx",
        index=False,
    )

    ranking_df = final_test_df.sort_values(
        ["f1_score", "auc", "pod_recall"],
        ascending=[False, False, False],
    )

    ranking_file = METRIC_DIR / "sequence_final_test_ranking_with_validation_threshold.csv"
    ranking_df.to_csv(ranking_file, index=False)
    ranking_df.to_excel(
        METRIC_DIR / "sequence_final_test_ranking_with_validation_threshold.xlsx",
        index=False,
    )

    report_file = REPORT_DIR / "sequence_threshold_from_validation_summary.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("SEQUENCE WINDOW THRESHOLD SELECTION FROM VALIDATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("Threshold dipilih menggunakan validation set tahun 2023.\n")
        f.write("Threshold terpilih kemudian diterapkan pada test set tahun 2024.\n\n")

        f.write("THRESHOLD RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        f.write(rec_df.to_string(index=False))
        f.write("\n\n")

        f.write("FINAL TEST METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(final_test_df.to_string(index=False))
        f.write("\n\n")

        f.write("RANKING\n")
        f.write("-" * 80 + "\n")
        f.write(ranking_df.to_string(index=False))

    print("\nSaved:")
    print(all_validation_file)
    print(rec_file)
    print(final_file)
    print(ranking_file)
    print(report_file)

    print("\nSequence threshold recommendations:")
    print(rec_df)

    print("\nSequence final test ranking:")
    print(ranking_df)


if __name__ == "__main__":
    main()