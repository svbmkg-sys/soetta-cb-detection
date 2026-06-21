import json
import random
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models, callbacks, optimizers

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

WINDOWS = [3, 5, 7]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "pod_recall": pod,
        "far": far,
        "f1_score": f1,
        "csi": csi,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def get_class_weight(y):
    classes = np.unique(y)

    if len(classes) < 2:
        return None

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    return {int(c): float(w) for c, w in zip(classes, weights)}


def make_sequences(X, y, years, times, window):
    X_seq = []
    y_seq = []
    year_seq = []
    time_seq = []

    for i in range(window - 1, len(X)):
        X_seq.append(X[i - window + 1 : i + 1])
        y_seq.append(y[i])
        year_seq.append(years[i])
        time_seq.append(times[i])

    return (
        np.array(X_seq),
        np.array(y_seq),
        np.array(year_seq),
        np.array(time_seq),
    )


def build_1dcnn(timesteps, n_features):
    model = models.Sequential(
        [
            layers.Input(shape=(timesteps, n_features)),
            layers.Conv1D(64, kernel_size=2, padding="causal", activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Conv1D(32, kernel_size=2, padding="causal", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.20):
    shortcut = x

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)

    return x


def build_tcn(timesteps, n_features):
    inputs = layers.Input(shape=(timesteps, n_features))

    x = residual_tcn_block(inputs, filters=64, kernel_size=2, dilation_rate=1)
    x = residual_tcn_block(x, filters=64, kernel_size=2, dilation_rate=2)
    x = residual_tcn_block(x, filters=32, kernel_size=2, dilation_rate=4)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def plot_training_history(history, model_name):
    hist = pd.DataFrame(history.history)

    plt.figure(figsize=(8, 5))
    plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")
    plt.title(f"Training Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{model_name}_training_loss.png", dpi=300)
    plt.close()

    if "auc" in hist.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["auc"], label="train_auc")
        if "val_auc" in hist:
            plt.plot(hist["val_auc"], label="val_auc")
        plt.title(f"Training AUC - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"{model_name}_training_auc.png", dpi=300)
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
    plt.savefig(FIGURE_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, model_name):
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
    plt.savefig(FIGURE_DIR / f"{model_name}_roc_curve.png", dpi=300)
    plt.close()


def train_model(model, X_train, y_train, X_val, y_val, model_name):
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=30,
        restore_best_weights=True,
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    class_weight = get_class_weight(y_train)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight,
        verbose=1,
    )

    model.save(MODEL_DIR / f"{model_name}.keras")
    plot_training_history(history, model_name)

    return model, history


def predict_save_and_evaluate(model, X, y, times, model_name, split_name):
    y_prob = model.predict(X).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    pred_df = pd.DataFrame(
        {
            "time": times,
            "y_true": y,
            "y_prob": y_prob,
            "y_pred": y_pred,
        }
    )

    pred_file = METRIC_DIR / f"{model_name}_{split_name}_predictions.csv"
    pred_df.to_csv(pred_file, index=False)

    metrics = calculate_metrics(y, y_prob, threshold=0.5)
    metrics["model"] = model_name
    metrics["split"] = split_name

    metrics_file = METRIC_DIR / f"{model_name}_{split_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    if split_name == "test":
        plot_confusion_matrix(y, y_prob, model_name)
        plot_roc_curve(y, y_prob, model_name)

    return metrics


def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("SEQUENCE WINDOW EXPERIMENT FOR 1D-CNN AND TCN")
    print("=" * 80)

    df = pd.read_csv(DATASET_FILE, parse_dates=["sounding_time_utc"])

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]

    df = df.sort_values("sounding_time_utc").reset_index(drop=True)
    df = df.dropna(subset=[MAIN_TARGET]).copy()
    df[MAIN_TARGET] = df[MAIN_TARGET].astype(int)

    train_mask = df["year"].isin(TRAIN_YEARS)
    val_mask = df["year"].isin(VAL_YEARS)
    test_mask = df["year"].isin(TEST_YEARS)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    print("Train:", train_df["sounding_time_utc"].min(), "to", train_df["sounding_time_utc"].max(), len(train_df))
    print("Val  :", val_df["sounding_time_utc"].min(), "to", val_df["sounding_time_utc"].max(), len(val_df))
    print("Test :", test_df["sounding_time_utc"].min(), "to", test_df["sounding_time_utc"].max(), len(test_df))

    # Fit preprocessing hanya pada train
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_raw = train_df[feature_cols]
    X_train = imputer.fit_transform(X_train_raw)
    X_train = scaler.fit_transform(X_train)

    X_all_raw = df[feature_cols]
    X_all = imputer.transform(X_all_raw)
    X_all = scaler.transform(X_all)

    y_all = df[MAIN_TARGET].to_numpy()
    years_all = df["year"].to_numpy()
    times_all = df["sounding_time_utc"].astype(str).to_numpy()

    joblib.dump(imputer, MODEL_DIR / "sequence_window_imputer.joblib")
    joblib.dump(scaler, MODEL_DIR / "sequence_window_scaler.joblib")

    all_test_metrics = []
    all_val_metrics = []
    sequence_summary = []

    for window in WINDOWS:
        print("\n" + "=" * 80)
        print(f"WINDOW = {window}")
        print("=" * 80)

        X_seq, y_seq, year_seq, time_seq = make_sequences(
            X_all,
            y_all,
            years_all,
            times_all,
            window=window,
        )

        seq_train_mask = np.isin(year_seq, TRAIN_YEARS)
        seq_val_mask = np.isin(year_seq, VAL_YEARS)
        seq_test_mask = np.isin(year_seq, TEST_YEARS)

        X_seq_train = X_seq[seq_train_mask]
        y_seq_train = y_seq[seq_train_mask]

        X_seq_val = X_seq[seq_val_mask]
        y_seq_val = y_seq[seq_val_mask]

        X_seq_test = X_seq[seq_test_mask]
        y_seq_test = y_seq[seq_test_mask]

        time_seq_val = time_seq[seq_val_mask]
        time_seq_test = time_seq[seq_test_mask]

        print("Sequence shape train:", X_seq_train.shape)
        print("Sequence shape val  :", X_seq_val.shape)
        print("Sequence shape test :", X_seq_test.shape)

        sequence_summary.append(
            {
                "window": window,
                "train_samples": len(X_seq_train),
                "val_samples": len(X_seq_val),
                "test_samples": len(X_seq_test),
                "timesteps": window,
                "n_features": X_seq_train.shape[2],
            }
        )

        # =====================
        # 1D-CNN
        # =====================
        cnn_model_name = f"cnn1d_w{window}"

        cnn = build_1dcnn(
            timesteps=X_seq_train.shape[1],
            n_features=X_seq_train.shape[2],
        )

        cnn, _ = train_model(
            cnn,
            X_seq_train,
            y_seq_train,
            X_seq_val,
            y_seq_val,
            cnn_model_name,
        )

        cnn_val_metrics = predict_save_and_evaluate(
            cnn,
            X_seq_val,
            y_seq_val,
            time_seq_val,
            cnn_model_name,
            "val",
        )

        cnn_test_metrics = predict_save_and_evaluate(
            cnn,
            X_seq_test,
            y_seq_test,
            time_seq_test,
            cnn_model_name,
            "test",
        )

        all_val_metrics.append(cnn_val_metrics)
        all_test_metrics.append(cnn_test_metrics)

        # =====================
        # TCN
        # =====================
        tcn_model_name = f"tcn_w{window}"

        tcn = build_tcn(
            timesteps=X_seq_train.shape[1],
            n_features=X_seq_train.shape[2],
        )

        tcn, _ = train_model(
            tcn,
            X_seq_train,
            y_seq_train,
            X_seq_val,
            y_seq_val,
            tcn_model_name,
        )

        tcn_val_metrics = predict_save_and_evaluate(
            tcn,
            X_seq_val,
            y_seq_val,
            time_seq_val,
            tcn_model_name,
            "val",
        )

        tcn_test_metrics = predict_save_and_evaluate(
            tcn,
            X_seq_test,
            y_seq_test,
            time_seq_test,
            tcn_model_name,
            "test",
        )

        all_val_metrics.append(tcn_val_metrics)
        all_test_metrics.append(tcn_test_metrics)

    val_metrics_df = pd.DataFrame(all_val_metrics)
    test_metrics_df = pd.DataFrame(all_test_metrics)
    sequence_summary_df = pd.DataFrame(sequence_summary)

    val_file = METRIC_DIR / "sequence_window_validation_metrics.csv"
    test_file = METRIC_DIR / "sequence_window_test_metrics.csv"
    summary_file = REPORT_DIR / "sequence_window_sample_summary.csv"

    val_metrics_df.to_csv(val_file, index=False)
    test_metrics_df.to_csv(test_file, index=False)
    sequence_summary_df.to_csv(summary_file, index=False)

    val_metrics_df.to_excel(METRIC_DIR / "sequence_window_validation_metrics.xlsx", index=False)
    test_metrics_df.to_excel(METRIC_DIR / "sequence_window_test_metrics.xlsx", index=False)
    sequence_summary_df.to_excel(REPORT_DIR / "sequence_window_sample_summary.xlsx", index=False)

    # Ranking by F1 and AUC
    ranking = test_metrics_df.sort_values(
        ["f1_score", "auc", "pod_recall"],
        ascending=[False, False, False],
    ).copy()

    ranking_file = METRIC_DIR / "sequence_window_test_ranking.csv"
    ranking.to_csv(ranking_file, index=False)
    ranking.to_excel(METRIC_DIR / "sequence_window_test_ranking.xlsx", index=False)

    # Simple report
    report_file = REPORT_DIR / "sequence_window_experiment_summary.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("SEQUENCE WINDOW EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Model yang diuji: 1D-CNN dan TCN\n")
        f.write(f"Sequence windows: {WINDOWS}\n")
        f.write(f"Target: {MAIN_TARGET}\n")
        f.write(f"Train years: {TRAIN_YEARS}\n")
        f.write(f"Validation years: {VAL_YEARS}\n")
        f.write(f"Test years: {TEST_YEARS}\n\n")

        f.write("SAMPLE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(sequence_summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("TEST METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(test_metrics_df.to_string(index=False))
        f.write("\n\n")

        f.write("RANKING BY F1, AUC, POD\n")
        f.write("-" * 80 + "\n")
        f.write(ranking.to_string(index=False))

    print("\nSaved:")
    print(val_file)
    print(test_file)
    print(summary_file)
    print(ranking_file)
    print(report_file)

    print("\nTest ranking:")
    print(ranking)


if __name__ == "__main__":
    main()