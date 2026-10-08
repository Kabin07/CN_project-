# ================================================================
# cn_project.py
# Quantized Lightweight CNN for IoT Malware Detection
# Dataset: N-BaIoT
# ================================================================

import os

# Must be set BEFORE TensorFlow is imported
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib


# ================================================================
# CONFIG
# ================================================================

from config import (
    DATA_MODE,
    DATA_ROOT,
    DEVICE_NAME,
    MAX_BENIGN_ROWS,
    MAX_MALWARE_ROWS,
    TEST_SIZE,
    VAL_SIZE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE
)


# ================================================================
# RANDOM SEED
# ================================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ================================================================
# PROJECT DIRECTORIES
# ================================================================

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / DATA_ROOT
DEVICE_DIR = DATA_DIR / DEVICE_NAME

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print(" QUANTIZED LIGHTWEIGHT CNN FOR IoT MALWARE DETECTION")
print("=" * 70)
print(f"TensorFlow version : {tf.__version__}")
print(f"Data mode          : {DATA_MODE}")
print(f"Dataset root       : {DATA_DIR}")
print(f"Device             : {DEVICE_NAME}")
print("=" * 70)


# ================================================================
# FIND BENIGN CSV
# ================================================================

def find_benign_file(device_dir):
    """
    Find benign_traffic.csv inside the selected device folder.
    """

    possible_files = [
        device_dir / "benign_traffic.csv",
        device_dir / "benign_traffic",
        device_dir / "benign.csv",
        device_dir / "benign"
    ]

    for file in possible_files:
        if file.exists():
            return file

    # Recursive search as backup
    csv_files = list(device_dir.rglob("*.csv"))

    for file in csv_files:

        if "benign" in file.name.lower():
            return file

    return None


# ================================================================
# FIND ATTACK CSV FILES
# ================================================================

def find_attack_csv_files(device_dir):
    """
    Find all attack CSV files inside:
        mirai_attacks/
        gafgyt_attacks/
    """

    attack_files = []

    attack_folders = [
        device_dir / "mirai_attacks",
        device_dir / "gafgyt_attacks"
    ]

    for folder in attack_folders:

        if not folder.exists():
            print(f"Warning: folder not found: {folder}")
            continue

        csv_files = list(folder.rglob("*.csv"))

        attack_files.extend(csv_files)

    return sorted(attack_files)


# ================================================================
# LOAD BENIGN DATA
# ================================================================

def load_benign_data(device_dir):

    print()
    print("-" * 70)
    print("LOADING BENIGN TRAFFIC")
    print("-" * 70)

    benign_file = find_benign_file(device_dir)

    if benign_file is None:

        raise FileNotFoundError(
            "\nbenign_traffic.csv was not found.\n\n"
            f"Searched inside:\n{device_dir}\n"
        )

    print(f"File: {benign_file}")

    try:

        df = pd.read_csv(
            benign_file,
            nrows=MAX_BENIGN_ROWS
        )

    except Exception as e:

        raise RuntimeError(
            f"\nCould not read benign CSV:\n"
            f"{benign_file}\n\n"
            f"Error: {e}"
        )

    if df.empty:
        raise ValueError("Benign CSV is empty.")

    # 0 = benign
    df["Label"] = 0

    print(
        f"Benign samples loaded: {len(df):,}"
    )

    print(
        f"Columns found: {len(df.columns)}"
    )

    return df


# ================================================================
# LOAD MALWARE DATA
# ================================================================

def load_malware_data(device_dir):

    print()
    print("-" * 70)
    print("SEARCHING FOR ATTACK DATA")
    print("-" * 70)

    attack_files = find_attack_csv_files(
        device_dir
    )

    if len(attack_files) == 0:

        raise FileNotFoundError(
            "\nNo attack CSV files were found.\n\n"
            f"Searched inside:\n{device_dir}\n\n"
            "Expected folders:\n"
            "mirai_attacks/\n"
            "gafgyt_attacks/\n"
        )

    print("\nAttack CSV files found:")

    for file in attack_files:
        print(f"  {file}")

    print()
    print("-" * 70)
    print("LOADING MALWARE TRAFFIC")
    print("-" * 70)

    malware_parts = []

    remaining_rows = MAX_MALWARE_ROWS

    for attack_file in attack_files:

        if remaining_rows <= 0:
            break

        print(
            f"\nReading: {attack_file.name}"
        )

        try:

            temp = pd.read_csv(
                attack_file,
                nrows=remaining_rows
            )

        except Exception as e:

            print(
                f"Could not read {attack_file}"
            )

            print(
                f"Reason: {e}"
            )

            continue

        if temp.empty:
            continue

        # 1 = malware
        temp["Label"] = 1

        malware_parts.append(temp)

        rows_loaded = len(temp)

        remaining_rows -= rows_loaded

        print(
            f"Loaded: {rows_loaded:,} rows"
        )

    if len(malware_parts) == 0:

        raise RuntimeError(
            "Attack CSV files were found, "
            "but none could be loaded."
        )

    malware = pd.concat(
        malware_parts,
        ignore_index=True
    )

    print()
    print(
        f"Total malware samples loaded: "
        f"{len(malware):,}"
    )

    return malware


# ================================================================
# LOAD N-BAIoT DATASET
# ================================================================

def load_nbaiot_data():

    print()
    print("=" * 70)
    print(" LOADING N-BAIoT DATASET")
    print("=" * 70)

    # ------------------------------------------------------------
    # Check dataset root
    # ------------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            "\nDataset root does not exist:\n"
            f"{DATA_DIR}\n\n"
            "Check DATA_ROOT in config.py."
        )

    # ------------------------------------------------------------
    # Check device folder
    # ------------------------------------------------------------

    if not DEVICE_DIR.exists():

        raise FileNotFoundError(
            "\nDevice folder does not exist:\n"
            f"{DEVICE_DIR}\n\n"
            "Check DEVICE_NAME in config.py."
        )

    print(
        f"\nDevice directory:\n{DEVICE_DIR}"
    )

    # ------------------------------------------------------------
    # Load benign
    # ------------------------------------------------------------

    benign = load_benign_data(
        DEVICE_DIR
    )

    # ------------------------------------------------------------
    # Load malware
    # ------------------------------------------------------------

    malware = load_malware_data(
        DEVICE_DIR
    )

    # ------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print("COMBINING DATA")
    print("-" * 70)

    df = pd.concat(
        [
            benign,
            malware
        ],
        ignore_index=True
    )

    # Shuffle
    df = df.sample(
        frac=1,
        random_state=SEED
    ).reset_index(
        drop=True
    )

    print(
        f"\nTotal samples : {len(df):,}"
    )

    print(
        f"Benign        : {(df['Label'] == 0).sum():,}"
    )

    print(
        f"Malware       : {(df['Label'] == 1).sum():,}"
    )

    return df


# ================================================================
# PREPROCESS DATA
# ================================================================

def preprocess_data(df):

    print()
    print("=" * 70)
    print(" PREPROCESSING DATA")
    print("=" * 70)

    df = df.copy()

    # ------------------------------------------------------------
    # Separate target
    # ------------------------------------------------------------

    if "Label" not in df.columns:

        raise ValueError(
            "Label column is missing."
        )

    y = df["Label"].astype(
        np.int32
    )

    X = df.drop(
        columns=["Label"]
    )

    # ------------------------------------------------------------
    # Keep only numeric columns
    # ------------------------------------------------------------

    X = X.select_dtypes(
        include=[np.number]
    )

    print(
        f"Numeric features found: "
        f"{X.shape[1]}"
    )

    # ------------------------------------------------------------
    # Replace infinity
    # ------------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ------------------------------------------------------------
    # Remove completely empty columns
    # ------------------------------------------------------------

    X = X.dropna(
        axis=1,
        how="all"
    )

    # ------------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------------

    for column in X.columns:

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0.0

        X[column] = X[column].fillna(
            median_value
        )

    # ------------------------------------------------------------
    # Convert to float32
    # ------------------------------------------------------------

    X = X.astype(
        np.float32
    )

    # ------------------------------------------------------------
    # Remove constant columns
    # ------------------------------------------------------------

    variance = X.var()

    useful_columns = variance[
        variance > 0
    ].index

    X = X[
        useful_columns
    ]

    print(
        f"Features after cleaning: "
        f"{X.shape[1]}"
    )

    print(
        f"Samples: {len(X):,}"
    )

    # ------------------------------------------------------------
    # Check classes
    # ------------------------------------------------------------

    print("\nClass distribution:")

    print(
        y.value_counts().sort_index()
    )

    if X.shape[1] < 3:

        raise ValueError(
            "Too few usable features."
        )

    return X, y


# ================================================================
# SAVE CLASS DISTRIBUTION GRAPH
# ================================================================

def save_class_distribution(y):

    benign_count = int(
        (y == 0).sum()
    )

    malware_count = int(
        (y == 1).sum()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        ["Benign", "Malware"],
        [
            benign_count,
            malware_count
        ]
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.title(
        "N-BaIoT Class Distribution"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ================================================================
# CREATE CNN MODEL
# ================================================================

def create_cnn_model(input_shape):

    print()
    print("=" * 70)
    print(" CREATING LIGHTWEIGHT CNN")
    print("=" * 70)

    model = keras.Sequential([

        layers.Input(
            shape=input_shape
        ),

        layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu"
        ),

        layers.MaxPooling1D(
            pool_size=2
        ),

        layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),

        layers.MaxPooling1D(
            pool_size=2
        ),

        layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu"
        ),

        layers.GlobalAveragePooling1D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.5
        ),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(

        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    model.summary()

    return model


# ================================================================
# TRAINING ACCURACY GRAPH
# ================================================================

def save_training_accuracy(history):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "training_accuracy.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ================================================================
# TRAINING LOSS GRAPH
# ================================================================

def save_training_loss(history):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "training_loss.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ================================================================
# CONFUSION MATRIX
# ================================================================

def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=[
            "Benign",
            "Malware"
        ],
        yticklabels=[
            "Benign",
            "Malware"
        ]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "N-BaIoT Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

    # Save numerical matrix
    matrix_df = pd.DataFrame(
        cm,
        index=[
            "Actual_Benign",
            "Actual_Malware"
        ],
        columns=[
            "Predicted_Benign",
            "Predicted_Malware"
        ]
    )

    matrix_df.to_csv(
        RESULTS_DIR /
        "confusion_matrix.csv"
    )


# ================================================================
# SAVE CLASSIFICATION REPORT
# ================================================================

def save_classification_report(
    y_true,
    y_pred
):

    report = classification_report(

        y_true,

        y_pred,

        target_names=[
            "Benign",
            "Malware"
        ],

        output_dict=True,

        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    output_path = (
        RESULTS_DIR /
        "classification_report.csv"
    )

    report_df.to_csv(
        output_path
    )

    print(
        f"Saved: {output_path}"
    )


# ================================================================
# SAVE TRAINING HISTORY
# ================================================================

def save_training_history(history):

    history_df = pd.DataFrame(
        history.history
    )

    output_path = (
        RESULTS_DIR /
        "training_history.csv"
    )

    history_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


# ================================================================
# SAVE DATASET SUMMARY
# ================================================================

def save_dataset_summary(
    df,
    X,
    y,
    X_train,
    X_val,
    X_test
):

    summary = pd.DataFrame([{

        "dataset":
            "N-BaIoT",

        "device":
            DEVICE_NAME,

        "total_samples":
            len(df),

        "benign_samples":
            int((y == 0).sum()),

        "malware_samples":
            int((y == 1).sum()),

        "features":
            X.shape[1],

        "training_samples":
            len(X_train),

        "validation_samples":
            len(X_val),

        "testing_samples":
            len(X_test)

    }])

    output_path = (
        RESULTS_DIR /
        "dataset_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


# ================================================================
# SAVE FINAL RESULTS
# ================================================================

def save_final_results(
    accuracy,
    precision,
    recall,
    f1,
    training_time,
    total_time,
    df,
    X,
    y,
    X_train,
    X_val,
    X_test
):

    results = pd.DataFrame([{

        "dataset":
            "N-BaIoT",

        "device":
            DEVICE_NAME,

        "total_samples":
            len(df),

        "benign_samples":
            int((y == 0).sum()),

        "malware_samples":
            int((y == 1).sum()),

        "features":
            X.shape[1],

        "training_samples":
            len(X_train),

        "validation_samples":
            len(X_val),

        "testing_samples":
            len(X_test),

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "training_time_seconds":
            float(training_time),

        "total_runtime_seconds":
            float(total_time)

    }])

    output_path = (
        RESULTS_DIR /
        "real_dataset_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    total_start = time.perf_counter()

    try:

        # ========================================================
        # CHECK DATA MODE
        # ========================================================

        if DATA_MODE.lower() != "nbaiot":

            raise ValueError(
                "DATA_MODE must be 'nbaiot'."
            )

        # ========================================================
        # LOAD DATA
        # ========================================================

        df = load_nbaiot_data()

        # ========================================================
        # PREPROCESS
        # ========================================================

        X, y = preprocess_data(
            df
        )

        # ========================================================
        # CLASS DISTRIBUTION
        # ========================================================

        save_class_distribution(
            y
        )

        # ========================================================
        # TRAIN / TEST SPLIT
        # ========================================================

        print()
        print("=" * 70)
        print(" TRAIN / TEST SPLIT")
        print("=" * 70)

        X_train_df, X_test_df, y_train, y_test = train_test_split(

            X,
            y,

            test_size=TEST_SIZE,

            random_state=SEED,

            stratify=y
        )

        # ========================================================
        # TRAIN / VALIDATION SPLIT
        # ========================================================

        X_train_df, X_val_df, y_train, y_val = train_test_split(

            X_train_df,

            y_train,

            test_size=VAL_SIZE,

            random_state=SEED,

            stratify=y_train
        )

        print(
            f"Training samples   : {len(X_train_df):,}"
        )

        print(
            f"Validation samples : {len(X_val_df):,}"
        )

        print(
            f"Testing samples    : {len(X_test_df):,}"
        )

        # ========================================================
        # SCALE FEATURES
        # ========================================================

        print()
        print("Scaling features...")

        scaler = StandardScaler()

        # IMPORTANT:
        # Fit scaler ONLY on training data.

        X_train = scaler.fit_transform(
            X_train_df
        )

        X_val = scaler.transform(
            X_val_df
        )

        X_test = scaler.transform(
            X_test_df
        )

        # ========================================================
        # SAVE SCALER
        # ========================================================

        scaler_path = (
            MODELS_DIR /
            "scaler.pkl"
        )

        joblib.dump(
            scaler,
            scaler_path
        )

        print(
            f"Scaler saved: {scaler_path}"
        )

        # ========================================================
        # FLOAT32
        # ========================================================

        X_train = X_train.astype(
            np.float32
        )

        X_val = X_val.astype(
            np.float32
        )

        X_test = X_test.astype(
            np.float32
        )

        # ========================================================
        # ADD CNN CHANNEL DIMENSION
        # ========================================================

        X_train = X_train[
            ..., np.newaxis
        ]

        X_val = X_val[
            ..., np.newaxis
        ]

        X_test = X_test[
            ..., np.newaxis
        ]

        print()
        print(
            f"CNN input shape: {X_train.shape}"
        )

        # ========================================================
        # SAVE DATASET SUMMARY
        # ========================================================

        save_dataset_summary(
            df,
            X,
            y,
            X_train,
            X_val,
            X_test
        )

        # ========================================================
        # CREATE MODEL
        # ========================================================

        model = create_cnn_model(
            X_train.shape[1:]
        )

        # ========================================================
        # TRAIN
        # ========================================================

        print()
        print("=" * 70)
        print(" TRAINING CNN")
        print("=" * 70)

        training_start = time.perf_counter()

        history = model.fit(

            X_train,

            y_train,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            verbose=1
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        print()
        print(
            f"Training time: "
            f"{training_time:.2f} seconds"
        )

        # ========================================================
        # SAVE MODEL
        # ========================================================

        model_path = (
            MODELS_DIR /
            "original_model.h5"
        )

        model.save(
            model_path
        )

        print(
            f"Model saved: {model_path}"
        )

        # ========================================================
        # EVALUATE MODEL
        # ========================================================

        print()
        print("=" * 70)
        print(" EVALUATING MODEL")
        print("=" * 70)

        test_loss, test_accuracy = model.evaluate(

            X_test,

            y_test,

            verbose=0
        )

        # ========================================================
        # PREDICTIONS
        # ========================================================

        probabilities = model.predict(

            X_test,

            verbose=0
        ).ravel()

        y_pred = (
            probabilities >= 0.5
        ).astype(
            np.int32
        )

        # ========================================================
        # METRICS
        # ========================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # ========================================================
        # PRINT RESULTS
        # ========================================================

        print()
        print("=" * 70)
        print(" REAL N-BAIoT RESULTS")
        print("=" * 70)

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1-score  : {f1:.4f}"
        )

        print()
        print("Classification Report:")
        print()

        print(
            classification_report(

                y_test,

                y_pred,

                target_names=[
                    "Benign",
                    "Malware"
                ],

                zero_division=0
            )
        )

        # ========================================================
        # SAVE REPORTS
        # ========================================================

        save_classification_report(
            y_test,
            y_pred
        )

        save_confusion_matrix(
            y_test,
            y_pred
        )

        save_training_history(
            history
        )

        # ========================================================
        # SAVE GRAPHS
        # ========================================================

        print()
        print("=" * 70)
        print(" GENERATING GRAPHS")
        print("=" * 70)

        save_training_accuracy(
            history
        )

        save_training_loss(
            history
        )

        # ========================================================
        # TOTAL TIME
        # ========================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        # ========================================================
        # SAVE FINAL RESULTS
        # ========================================================

        save_final_results(

            accuracy,

            precision,

            recall,

            f1,

            training_time,

            total_time,

            df,

            X,

            y,

            X_train,

            X_val,

            X_test
        )

        # ========================================================
        # FINAL MESSAGE
        # ========================================================

        print()
        print("=" * 70)
        print(" CNN EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print()
        print("Generated files:")
        print()

        print(
            "Model:"
        )

        print(
            "  models/original_model.h5"
        )

        print(
            "  models/scaler.pkl"
        )

        print()
        print(
            "Results:"
        )

        print(
            "  results/real_dataset_results.csv"
        )

        print(
            "  results/dataset_summary.csv"
        )

        print(
            "  results/classification_report.csv"
        )

        print(
            "  results/confusion_matrix.csv"
        )

        print(
            "  results/training_history.csv"
        )

        print()
        print(
            "Graphs:"
        )

        print(
            "  plots/class_distribution.png"
        )

        print(
            "  plots/training_accuracy.png"
        )

        print(
            "  plots/training_loss.png"
        )

        print(
            "  plots/confusion_matrix.png"
        )

        print()
        print(
            "Next stage: INT8 quantization and "
            "FP32 vs INT8 comparison."
        )

        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print(" PROJECT FAILED")
        print("=" * 70)

        print()
        print("Error:")
        print(e)

        print()
        print(
            "Check the error above before changing "
            "any project files."
        )

        print("=" * 70)

        raise


# ================================================================
# PROGRAM ENTRY
# ================================================================

if __name__ == "__main__":
    main()