import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _config import (
    PROCESSED_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    MODEL_DIR,
    REPORT_DIR,
    MAIN_TARGET,
    TRAIN_YEARS,
    VAL_YEARS,
    TEST_YEARS,
    RANDOM_SEED,
)

warnings.filterwarnings("ignore")


DATASET_FILE = PROCESSED_DIR / "cb_dataset_tabular.csv"
METADATA_FILE = PROCESSED_DIR / "dataset_metadata.json"

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)


def calculate_metrics(y_true, y_prob, threshold=0.5):
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
        "threshold": float(threshold),
    }


def choose_threshold_from_validation(y_val, p_val):
    rows = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(y_val, p_val, threshold=threshold)
        rows.append(metrics)

    result_df = pd.DataFrame(rows)

    best_f1 = result_df.sort_values(
        ["f1_score", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    best_csi = result_df.sort_values(
        ["csi", "pod_recall", "far"],
        ascending=[False, False, True],
    ).iloc[0]

    candidates = result_df[result_df["pod_recall"] >= 0.75].copy()

    if len(candidates) > 0:
        operational = candidates.sort_values(
            ["far", "f1_score", "accuracy"],
            ascending=[True, False, False],
        ).iloc[0]
        operational_note = "POD validation >= 0.75, FAR minimum"
    else:
        operational = best_f1
        operational_note = "Tidak ada threshold dengan POD validation >= 0.75; menggunakan F1 maksimum"

    return result_df, {
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


def plot_roc(y_true, y_prob, model_name):
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / POD")
    plt.legend()
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(FIGURE_DIR / f"baseline_{safe_name}_roc_curve.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_prob, model_name, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(FIGURE_DIR / f"baseline_{safe_name}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_feature_importance_random_forest(model, feature_cols):
    rf = model.named_steps["model"]

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_file = REPORT_DIR / "random_forest_feature_importance.csv"
    importance_df.to_csv(importance_file, index=False)

    top_n = min(20, len(importance_df))
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 7))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.title("Top Feature Importance - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "random_forest_feature_importance_top20.png", dpi=300)
    plt.close()

    print("Saved:", importance_file)

    return importance_df


def main():
    print("=" * 80)
    print("TRAINING BASELINE MODELS")
    print("=" * 80)

    df = pd.read_csv(DATASET_FILE, parse_dates=["sounding_time_utc"])

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]

    df = df.sort_values("sounding_time_utc").reset_index(drop=True)
    df = df.dropna(subset=[MAIN_TARGET]).copy()
    df[MAIN_TARGET] = df[MAIN_TARGET].astype(int)

    train_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    val_df = df[df["year"].isin(VAL_YEARS)].copy()
    test_df = df[df["year"].isin(TEST_YEARS)].copy()

    X_train = train_df[feature_cols]
    y_train = train_df[MAIN_TARGET].to_numpy()

    X_val = val_df[feature_cols]
    y_val = val_df[MAIN_TARGET].to_numpy()

    X_test = test_df[feature_cols]
    y_test = test_df[MAIN_TARGET].to_numpy()

    print("Train:", train_df["sounding_time_utc"].min(), "to", train_df["sounding_time_utc"].max(), len(train_df))
    print("Val  :", val_df["sounding_time_utc"].min(), "to", val_df["sounding_time_utc"].max(), len(val_df))
    print("Test :", test_df["sounding_time_utc"].min(), "to", test_df["sounding_time_utc"].max(), len(test_df))

    print("\nTarget distribution:")
    print("Train:")
    print(pd.Series(y_train).value_counts())
    print("Validation:")
    print(pd.Series(y_val).value_counts())
    print("Test:")
    print(pd.Series(y_test).value_counts())

    models = {
        "Dummy Stratified": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    DummyClassifier(
                        strategy="stratified",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    default_rows = []
    validation_threshold_rows = []
    final_test_rows = []

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(model_name)
        print("=" * 80)

        safe_name = model_name.lower().replace(" ", "_")

        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_DIR / f"baseline_{safe_name}.joblib")

        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        # Save predictions
        val_pred_df = pd.DataFrame(
            {
                "time": val_df["sounding_time_utc"].astype(str).to_numpy(),
                "y_true": y_val,
                "y_prob": p_val,
                "y_pred_default_0_50": (p_val >= 0.5).astype(int),
            }
        )

        test_pred_df = pd.DataFrame(
            {
                "time": test_df["sounding_time_utc"].astype(str).to_numpy(),
                "y_true": y_test,
                "y_prob": p_test,
                "y_pred_default_0_50": (p_test >= 0.5).astype(int),
            }
        )

        val_pred_df.to_csv(METRIC_DIR / f"baseline_{safe_name}_val_predictions.csv", index=False)
        test_pred_df.to_csv(METRIC_DIR / f"baseline_{safe_name}_test_predictions.csv", index=False)

        # Default threshold test metrics
        default_metrics = calculate_metrics(y_test, p_test, threshold=0.5)
        default_metrics["model"] = model_name
        default_metrics["threshold_source"] = "default_0_50"
        default_rows.append(default_metrics)

        # Threshold from validation
        val_threshold_df, rec = choose_threshold_from_validation(y_val, p_val)
        val_threshold_df["model"] = model_name

        val_threshold_df.to_csv(
            METRIC_DIR / f"baseline_{safe_name}_validation_threshold_analysis.csv",
            index=False,
        )

        rec["model"] = model_name
        validation_threshold_rows.append(rec)

        threshold_options = {
            "best_f1_from_validation": rec["best_f1_threshold"],
            "best_csi_from_validation": rec["best_csi_threshold"],
            "operational_from_validation": rec["operational_threshold"],
        }

        for threshold_source, threshold_value in threshold_options.items():
            test_metrics = calculate_metrics(y_test, p_test, threshold=threshold_value)
            test_metrics["model"] = model_name
            test_metrics["threshold_source"] = threshold_source
            final_test_rows.append(test_metrics)

        plot_roc(y_test, p_test, model_name)
        plot_confusion_matrix(y_test, p_test, model_name, threshold=0.5)

        if model_name == "Random Forest":
            plot_feature_importance_random_forest(model, feature_cols)

        print("Default threshold test metrics:")
        print(default_metrics)

    default_df = pd.DataFrame(default_rows)
    validation_threshold_df = pd.DataFrame(validation_threshold_rows)
    final_test_df = pd.DataFrame(final_test_rows)

    # Tambahkan default ke final test comparison
    final_with_default_df = pd.concat([default_df, final_test_df], ignore_index=True)

    cols = [
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

    default_df = default_df[cols]
    final_with_default_df = final_with_default_df[cols]

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

    validation_threshold_df = validation_threshold_df[rec_cols]

    default_csv = METRIC_DIR / "baseline_default_test_metrics.csv"
    threshold_rec_csv = METRIC_DIR / "baseline_threshold_recommendations_from_validation.csv"
    final_csv = METRIC_DIR / "baseline_final_test_metrics_with_validation_threshold.csv"

    default_df.to_csv(default_csv, index=False)
    validation_threshold_df.to_csv(threshold_rec_csv, index=False)
    final_with_default_df.to_csv(final_csv, index=False)

    default_df.to_excel(METRIC_DIR / "baseline_default_test_metrics.xlsx", index=False)
    validation_threshold_df.to_excel(
        METRIC_DIR / "baseline_threshold_recommendations_from_validation.xlsx",
        index=False,
    )
    final_with_default_df.to_excel(
        METRIC_DIR / "baseline_final_test_metrics_with_validation_threshold.xlsx",
        index=False,
    )

    # Ranking
    ranking_df = final_with_default_df.sort_values(
        ["f1_score", "auc", "pod_recall"],
        ascending=[False, False, False],
    ).copy()

    ranking_csv = METRIC_DIR / "baseline_model_ranking.csv"
    ranking_df.to_csv(ranking_csv, index=False)
    ranking_df.to_excel(METRIC_DIR / "baseline_model_ranking.xlsx", index=False)

    report_file = REPORT_DIR / "baseline_model_summary.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("BASELINE MODEL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Target: {MAIN_TARGET}\n")
        f.write(f"Train years: {TRAIN_YEARS}\n")
        f.write(f"Validation years: {VAL_YEARS}\n")
        f.write(f"Test years: {TEST_YEARS}\n\n")

        f.write("BASELINE DEFAULT TEST METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(default_df.to_string(index=False))
        f.write("\n\n")

        f.write("THRESHOLD RECOMMENDATIONS FROM VALIDATION\n")
        f.write("-" * 80 + "\n")
        f.write(validation_threshold_df.to_string(index=False))
        f.write("\n\n")

        f.write("FINAL TEST METRICS WITH VALIDATION THRESHOLD\n")
        f.write("-" * 80 + "\n")
        f.write(final_with_default_df.to_string(index=False))
        f.write("\n\n")

        f.write("RANKING\n")
        f.write("-" * 80 + "\n")
        f.write(ranking_df.to_string(index=False))

    print("\nSaved:")
    print(default_csv)
    print(threshold_rec_csv)
    print(final_csv)
    print(ranking_csv)
    print(report_file)

    print("\nBaseline ranking:")
    print(ranking_df)


if __name__ == "__main__":
    main()