import pandas as pd

from _config import METRIC_DIR, REPORT_DIR


OUTPUT_CSV = METRIC_DIR / "final_model_comparison_selected.csv"
OUTPUT_XLSX = METRIC_DIR / "final_model_comparison_selected.xlsx"
OUTPUT_REPORT = REPORT_DIR / "final_model_comparison_summary.txt"


def load_baseline_results():
    baseline_file = METRIC_DIR / "baseline_model_ranking.csv"
    df = pd.read_csv(baseline_file)

    selected = []

    # Dummy Stratified default
    dummy = df[
        (df["model"] == "Dummy Stratified")
        & (df["threshold_source"] == "default_0_50")
    ].copy()

    if len(dummy) > 0:
        row = dummy.iloc[0].to_dict()
        row["group"] = "Baseline"
        row["configuration"] = "Dummy stratified, threshold 0.50"
        selected.append(row)

    # Logistic Regression best F1 from validation
    logreg = df[
        (df["model"] == "Logistic Regression")
        & (df["threshold_source"] == "best_f1_from_validation")
    ].copy()

    if len(logreg) > 0:
        row = logreg.iloc[0].to_dict()
        row["group"] = "Baseline"
        row["configuration"] = "Logistic Regression, threshold selected from validation"
        selected.append(row)

    # Random Forest best F1 from validation
    rf = df[
        (df["model"] == "Random Forest")
        & (df["threshold_source"] == "best_f1_from_validation")
    ].copy()

    if len(rf) > 0:
        row = rf.iloc[0].to_dict()
        row["group"] = "Baseline"
        row["configuration"] = "Random Forest, threshold selected from validation"
        selected.append(row)

    return selected


def load_deep_learning_results():
    dl_file = METRIC_DIR / "final_test_metrics_with_validation_threshold.csv"
    df = pd.read_csv(dl_file)

    selected = []

    # DNN best F1 from validation
    dnn = df[
        (df["model"] == "DNN")
        & (df["threshold_source"] == "best_f1_from_validation")
    ].copy()

    if len(dnn) > 0:
        row = dnn.iloc[0].to_dict()
        row["group"] = "Deep Learning"
        row["configuration"] = "DNN tabular, threshold selected from validation"
        selected.append(row)

    return selected


def load_sequence_results():
    seq_file = METRIC_DIR / "sequence_final_test_metrics_with_validation_threshold.csv"
    df = pd.read_csv(seq_file)

    selected = []

    # 1D-CNN final: window 7 best F1 from validation
    cnn = df[
        (df["model"] == "1D-CNN_w7")
        & (df["threshold_source"] == "best_f1_from_validation")
    ].copy()

    if len(cnn) > 0:
        row = cnn.iloc[0].to_dict()
        row["group"] = "Deep Learning"
        row["configuration"] = "1D-CNN, sequence window 7, threshold selected from validation"
        selected.append(row)

    # TCN final: window 5 best F1 from validation
    tcn = df[
        (df["model"] == "TCN_w5")
        & (df["threshold_source"] == "best_f1_from_validation")
    ].copy()

    if len(tcn) > 0:
        row = tcn.iloc[0].to_dict()
        row["group"] = "Deep Learning"
        row["configuration"] = "TCN, sequence window 5, threshold selected from validation"
        selected.append(row)

    return selected


def main():
    print("=" * 80)
    print("BUILD FINAL MODEL COMPARISON TABLE")
    print("=" * 80)

    rows = []
    rows.extend(load_baseline_results())
    rows.extend(load_deep_learning_results())
    rows.extend(load_sequence_results())

    final_df = pd.DataFrame(rows)

    final_cols = [
        "group",
        "model",
        "configuration",
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

    final_df = final_df[final_cols]

    # Urutan tampil yang lebih enak dibaca
    model_order = {
        "Dummy Stratified": 1,
        "Logistic Regression": 2,
        "Random Forest": 3,
        "DNN": 4,
        "1D-CNN_w7": 5,
        "TCN_w5": 6,
    }

    final_df["model_order"] = final_df["model"].map(model_order)
    final_df = final_df.sort_values("model_order").drop(columns=["model_order"])

    # Ranking tambahan berdasarkan F1-score
    ranking_df = final_df.sort_values(
        ["f1_score", "auc", "pod_recall"],
        ascending=[False, False, False],
    ).copy()

    final_df.to_csv(OUTPUT_CSV, index=False)
    final_df.to_excel(OUTPUT_XLSX, index=False)

    ranking_csv = METRIC_DIR / "final_model_comparison_ranking.csv"
    ranking_xlsx = METRIC_DIR / "final_model_comparison_ranking.xlsx"

    ranking_df.to_csv(ranking_csv, index=False)
    ranking_df.to_excel(ranking_xlsx, index=False)

    best_f1 = ranking_df.iloc[0]
    best_auc = final_df.sort_values("auc", ascending=False).iloc[0]
    best_pod = final_df.sort_values("pod_recall", ascending=False).iloc[0]
    best_far = final_df.sort_values("far", ascending=True).iloc[0]

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("FINAL MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("Selected final models:\n")
        f.write(final_df.to_string(index=False))
        f.write("\n\n")

        f.write("Ranking by F1-score, AUC, and POD:\n")
        f.write(ranking_df.to_string(index=False))
        f.write("\n\n")

        f.write("Key findings:\n")
        f.write(f"- Best F1-score: {best_f1['model']} with F1 = {best_f1['f1_score']:.4f}\n")
        f.write(f"- Best AUC: {best_auc['model']} with AUC = {best_auc['auc']:.4f}\n")
        f.write(f"- Best POD/Recall: {best_pod['model']} with POD = {best_pod['pod_recall']:.4f}\n")
        f.write(f"- Lowest FAR: {best_far['model']} with FAR = {best_far['far']:.4f}\n")

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_XLSX)
    print(ranking_csv)
    print(ranking_xlsx)
    print(OUTPUT_REPORT)

    print("\nFinal selected model comparison:")
    print(final_df)

    print("\nRanking:")
    print(ranking_df)


if __name__ == "__main__":
    main()