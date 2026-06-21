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
    SEQUENCE_WINDOW,
)

warnings.filterwarnings("ignore")


DATASET_FILE = PROCESSED_DIR / "cb_dataset_tabular.csv"
METADATA_FILE = PROCESSED_DIR / "dataset_metadata.json"


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

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    return {
        "accuracy": accuracy,
        "pod_recall": pod,
        "far": far,
        "f1_score": f1,
        "auc": auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


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


def build_dnn(input_dim):
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.30),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
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


def predict_and_save(model, X, y, times, model_name, split_name):
    y_prob = model.predict(X).ravel()

    metrics = calculate_metrics(y, y_prob, threshold=0.5)

    metrics_file = METRIC_DIR / f"{model_name}_{split_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    pred_df = pd.DataFrame(
        {
            "time": times,
            "y_true": y,
            "y_prob": y_prob,
            "y_pred": (y_prob >= 0.5).astype(int),
        }
    )

    pred_file = METRIC_DIR / f"{model_name}_{split_name}_predictions.csv"
    pred_df.to_csv(pred_file, index=False)

    if split_name == "test":
        plot_confusion_matrix(y, y_prob, model_name)
        plot_roc_curve(y, y_prob, model_name)

    print(f"\n{model_name} {split_name} metrics at threshold 0.5:")
    print(metrics)

    return metrics


def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("TRAINING DNN, 1D-CNN, AND TCN")
    print("=" * 80)

    df = pd.read_csv(DATASET_FILE, parse_dates=["sounding_time_utc"])

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_columns"]

    df = df.sort_values("sounding_time_utc").reset_index(drop=True)

    # Pastikan target valid
    df = df.dropna(subset=[MAIN_TARGET]).copy()
    df[MAIN_TARGET] = df[MAIN_TARGET].astype(int)

    train_mask = df["year"].isin(TRAIN_YEARS)
    val_mask = df["year"].isin(VAL_YEARS)
    test_mask = df["year"].isin(TEST_YEARS)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    print("Train period:", train_df["sounding_time_utc"].min(), "to", train_df["sounding_time_utc"].max(), len(train_df))
    print("Val period:", val_df["sounding_time_utc"].min(), "to", val_df["sounding_time_utc"].max(), len(val_df))
    print("Test period:", test_df["sounding_time_utc"].min(), "to", test_df["sounding_time_utc"].max(), len(test_df))

    print("\nTarget distribution:")
    print("Train:")
    print(train_df[MAIN_TARGET].value_counts())
    print("Validation:")
    print(val_df[MAIN_TARGET].value_counts())
    print("Test:")
    print(test_df[MAIN_TARGET].value_counts())

    X_train_raw = train_df[feature_cols]
    X_val_raw = val_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    y_train = train_df[MAIN_TARGET].to_numpy()
    y_val = val_df[MAIN_TARGET].to_numpy()
    y_test = test_df[MAIN_TARGET].to_numpy()

    # Preprocessing fit hanya pada train untuk menghindari data leakage
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train_raw)
    X_train = scaler.fit_transform(X_train)

    X_val = imputer.transform(X_val_raw)
    X_val = scaler.transform(X_val)

    X_test = imputer.transform(X_test_raw)
    X_test = scaler.transform(X_test)

    joblib.dump(imputer, MODEL_DIR / "imputer.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    all_metrics = []

    # =========================
    # DNN
    # =========================

    dnn = build_dnn(input_dim=X_train.shape[1])
    dnn, _ = train_model(dnn, X_train, y_train, X_val, y_val, "dnn")

    dnn_val_metrics = predict_and_save(
        dnn,
        X_val,
        y_val,
        val_df["sounding_time_utc"].astype(str).to_numpy(),
        "dnn",
        "val",
        )
    
    dnn_metrics = predict_and_save(
        dnn,
        X_test,
        y_test,
        test_df["sounding_time_utc"].astype(str).to_numpy(),
        "dnn",
        "test",
        )
    
    dnn_metrics["model"] = "DNN"
    all_metrics.append(dnn_metrics)

    # =========================
    # Sequence data for 1D-CNN and TCN
    # =========================

    X_all_raw = df[feature_cols]
    X_all = imputer.transform(X_all_raw)
    X_all = scaler.transform(X_all)

    y_all = df[MAIN_TARGET].to_numpy()
    years_all = df["year"].to_numpy()
    times_all = df["sounding_time_utc"].astype(str).to_numpy()

    X_seq, y_seq, year_seq, time_seq = make_sequences(
        X_all,
        y_all,
        years_all,
        times_all,
        window=SEQUENCE_WINDOW,
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
    time_seq_test = time_seq[seq_test_mask]

    print("\nSequence shapes:")
    print("Train:", X_seq_train.shape)
    print("Val:", X_seq_val.shape)
    print("Test:", X_seq_test.shape)

    # =========================
    # 1D-CNN
    # =========================

    cnn = build_1dcnn(
        timesteps=X_seq_train.shape[1],
        n_features=X_seq_train.shape[2],
    )
    cnn, _ = train_model(cnn, X_seq_train, y_seq_train, X_seq_val, y_seq_val, "cnn1d")
    
    time_seq_val = time_seq[seq_val_mask]
    
    cnn_val_metrics = predict_and_save(
        cnn,
        X_seq_val,
        y_seq_val,
        time_seq_val,
        "cnn1d",
        "val",
        )
    
    cnn_metrics = predict_and_save(
        cnn,
        X_seq_test,
        y_seq_test,
        time_seq_test,
        "cnn1d",
        "test",
        )
    
    cnn_metrics["model"] = "1D-CNN"
    all_metrics.append(cnn_metrics)

    # =========================
    # TCN
    # =========================

    tcn = build_tcn(
        timesteps=X_seq_train.shape[1],
        n_features=X_seq_train.shape[2],
    )
    tcn, _ = train_model(tcn, X_seq_train, y_seq_train, X_seq_val, y_seq_val, "tcn")

    tcn_val_metrics = predict_and_save(
        tcn,
        X_seq_val,
        y_seq_val,
        time_seq_val,
        "tcn",
        "val",)
    
    tcn_metrics = predict_and_save(
        tcn,
        X_seq_test,
        y_seq_test,
        time_seq_test,
        "tcn",
        "test",)
    
    tcn_metrics["model"] = "TCN"
    all_metrics.append(tcn_metrics)

    # =========================
    # Save comparison table
    # =========================

    metrics_df = pd.DataFrame(all_metrics)

    cols = [
        "model",
        "accuracy",
        "pod_recall",
        "far",
        "f1_score",
        "auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "threshold",
    ]
    metrics_df = metrics_df[cols]

    comparison_file = METRIC_DIR / "model_comparison_metrics.csv"
    comparison_xlsx = METRIC_DIR / "model_comparison_metrics.xlsx"

    metrics_df.to_csv(comparison_file, index=False)
    metrics_df.to_excel(comparison_xlsx, index=False)

    print("\nModel comparison:")
    print(metrics_df)

    print("\nSaved:")
    print(comparison_file)
    print(comparison_xlsx)

    # Simple text report
    report_file = REPORT_DIR / "modeling_summary.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Target label: {MAIN_TARGET}\n")
        f.write(f"Train years: {TRAIN_YEARS}\n")
        f.write(f"Validation years: {VAL_YEARS}\n")
        f.write(f"Test years: {TEST_YEARS}\n")
        f.write(f"Sequence window: {SEQUENCE_WINDOW}\n\n")
        f.write(metrics_df.to_string(index=False))

    print(report_file)


if __name__ == "__main__":
    main()