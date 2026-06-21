import pandas as pd

from _config import METRIC_DIR, REPORT_DIR, PROCESSED_DIR


FINAL_MODEL_FILE = METRIC_DIR / "final_model_comparison_selected.csv"
FEATURE_IMPORTANCE_FILE = REPORT_DIR / "random_forest_feature_importance.csv"
EDA_SUMMARY_FILE = REPORT_DIR / "eda_summary.txt"
METADATA_FILE = PROCESSED_DIR / "dataset_metadata.json"


def percent(x):
    return round(float(x) * 100, 2)


def prepare_model_table():
    df = pd.read_csv(FINAL_MODEL_FILE)

    table = pd.DataFrame({
        "Kelompok": df["group"],
        "Model": df["model"],
        "Konfigurasi": df["configuration"],
        "Threshold": df["threshold"],
        "Accuracy (%)": df["accuracy"].apply(percent),
        "POD / Recall (%)": df["pod_recall"].apply(percent),
        "FAR (%)": df["far"].apply(percent),
        "F1-score (%)": df["f1_score"].apply(percent),
        "CSI (%)": df["csi"].apply(percent),
        "AUC (%)": df["auc"].apply(percent),
        "TN": df["tn"],
        "FP": df["fp"],
        "FN": df["fn"],
        "TP": df["tp"],
    })

    output_csv = REPORT_DIR / "tabel_final_perbandingan_model.csv"
    output_xlsx = REPORT_DIR / "tabel_final_perbandingan_model.xlsx"

    table.to_csv(output_csv, index=False)
    table.to_excel(output_xlsx, index=False)

    print("Saved:", output_csv)
    print("Saved:", output_xlsx)

    return table


def prepare_model_ranking_table():
    df = pd.read_csv(FINAL_MODEL_FILE)

    ranking = df.sort_values(
        ["f1_score", "auc", "pod_recall"],
        ascending=[False, False, False],
    ).copy()

    table = pd.DataFrame({
        "Peringkat": range(1, len(ranking) + 1),
        "Model": ranking["model"],
        "Threshold": ranking["threshold"],
        "Accuracy (%)": ranking["accuracy"].apply(percent),
        "POD / Recall (%)": ranking["pod_recall"].apply(percent),
        "FAR (%)": ranking["far"].apply(percent),
        "F1-score (%)": ranking["f1_score"].apply(percent),
        "CSI (%)": ranking["csi"].apply(percent),
        "AUC (%)": ranking["auc"].apply(percent),
    })

    output_csv = REPORT_DIR / "tabel_ranking_model_berdasarkan_f1.csv"
    output_xlsx = REPORT_DIR / "tabel_ranking_model_berdasarkan_f1.xlsx"

    table.to_csv(output_csv, index=False)
    table.to_excel(output_xlsx, index=False)

    print("Saved:", output_csv)
    print("Saved:", output_xlsx)

    return table


def prepare_feature_importance_table(top_n=15):
    df = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    df = df.head(top_n).copy()

    table = pd.DataFrame({
        "Peringkat": range(1, len(df) + 1),
        "Fitur": df["feature"],
        "Importance (%)": df["importance"].apply(percent),
    })

    output_csv = REPORT_DIR / f"tabel_top{top_n}_feature_importance_random_forest.csv"
    output_xlsx = REPORT_DIR / f"tabel_top{top_n}_feature_importance_random_forest.xlsx"

    table.to_csv(output_csv, index=False)
    table.to_excel(output_xlsx, index=False)

    print("Saved:", output_csv)
    print("Saved:", output_xlsx)

    return table


def main():
    print("=" * 80)
    print("PREPARE THESIS TABLES")
    print("=" * 80)

    model_table = prepare_model_table()
    ranking_table = prepare_model_ranking_table()
    feature_table = prepare_feature_importance_table(top_n=15)

    summary_file = REPORT_DIR / "ringkasan_hasil_final_untuk_tesis.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("RINGKASAN HASIL FINAL UNTUK TESIS\n")
        f.write("=" * 80 + "\n\n")

        f.write("TABEL FINAL PERBANDINGAN MODEL\n")
        f.write("-" * 80 + "\n")
        f.write(model_table.to_string(index=False))
        f.write("\n\n")

        f.write("RANKING MODEL BERDASARKAN F1-SCORE\n")
        f.write("-" * 80 + "\n")
        f.write(ranking_table.to_string(index=False))
        f.write("\n\n")

        f.write("TOP 15 FEATURE IMPORTANCE RANDOM FOREST\n")
        f.write("-" * 80 + "\n")
        f.write(feature_table.to_string(index=False))
        f.write("\n")

    print("Saved:", summary_file)


if __name__ == "__main__":
    main()