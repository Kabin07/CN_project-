import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import RESULTS_DIR, PLOTS_DIR

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RESULTS_DIR / "detailed_metrics.csv")

    for column, title, ylabel, filename in [
        ("accuracy", "Accuracy Comparison", "Accuracy", "model_accuracy_comparison.png"),
        ("size_kb", "Model Size Comparison", "Size (KB)", "model_size_comparison.png"),
        ("inference_ms", "Inference Time Comparison", "Time (ms/sample)", "inference_time_comparison.png")
    ]:
        plt.figure(figsize=(8, 5))
        plt.bar(df["model"], df[column])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=200)
        plt.close()

    predictions = pd.read_csv(RESULTS_DIR / "predictions.csv")

    for column, filename, title in [
        ("keras_fp32", "confusion_matrix_fp32.png", "Keras FP32 Confusion Matrix"),
        ("tflite_int8", "confusion_matrix_int8.png", "TFLite INT8 Confusion Matrix")
    ]:
        cm = confusion_matrix(predictions["actual"], predictions[column])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malware"])
        disp.plot()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=200)
        plt.close()

    history_path = RESULTS_DIR / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plt.figure(figsize=(8, 5))
        plt.plot(history["accuracy"], label="Training")
        plt.plot(history["val_accuracy"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "training_accuracy.png", dpi=200)
        plt.close()

    print("Graphs saved to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
