# Re-run evaluation using the saved models and processed test set.
# The main script already performs evaluation; this file is provided so the
# experiment can be repeated without retraining.

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import MODEL_DIR, PROCESSED_DIR, RESULTS_DIR

def runner(path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    def predict(x):
        x = x[np.newaxis].astype(np.float32)
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            x = np.round(x / scale + zero).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            scale, zero = out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return float(y.reshape(-1)[0])

    return predict

def main():
    X = np.load(PROCESSED_DIR / "X_test.npy")
    y = np.load(PROCESSED_DIR / "y_test.npy").astype(np.int8)

    rows = []
    for name, path in [
        ("TFLite FP32", MODEL_DIR / "fp32_model.tflite"),
        ("TFLite INT8", MODEL_DIR / "quantized_model_int8.tflite")
    ]:
        predict = runner(path)
        p = np.array([predict(x) for x in X])
        pred = (p >= 0.5).astype(np.int8)

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "size_kb": path.stat().st_size / 1024
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "reevaluated_tflite_metrics.csv", index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
