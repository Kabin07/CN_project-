# Quantized Lightweight CNN for IoT Malware Detection

Complete local Windows implementation based on the supplied project implementation.

The project supports:
- Real CSV network-traffic data (recommended)
- Synthetic fallback data for testing the pipeline
- StandardScaler preprocessing
- 1-D lightweight CNN
- Binary benign/malware classification
- FP32 Keras model
- FP32 TensorFlow Lite model
- Full INT8 post-training quantization
- Accuracy, precision, recall and F1
- Model size in KB
- Inference time in milliseconds
- Size reduction and speed improvement
- Confusion matrices
- Training history graphs
- CSV result export

## 1. Project structure

IoT_Malware_Quantized_CNN_Project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── plots/
├── results/
├── config.py
├── cn_project.py
├── evaluate.py
├── make_graphs.py
├── run_project.ps1
├── requirements.txt
└── README.md

## 2. Recommended environment

Windows + PowerShell + Python 3.10.

Check:

python --version

Create environment:

python -m venv .venv

Activate:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Install:

python -m pip install --upgrade pip
pip install -r requirements.txt

## 3. Real N-BaIoT data

Download N-BaIoT from the UCI Machine Learning Repository.

For the supplied pipeline, put CSV files in:

data/raw/

You can configure the exact filenames in config.py.

The loader accepts either:
A) one CSV containing a label column, or
B) separate benign and malware CSV files.

For a single labelled CSV, configure:

DATA_MODE = "single_csv"
SINGLE_CSV = "n_baiot.csv"
LABEL_COLUMN = "Label"

For separate files:

DATA_MODE = "two_csv"
BENIGN_FILE = "benign_traffic.csv"
MALWARE_FILE = "mirai_attacks.csv"

The script automatically keeps numeric feature columns and removes common non-feature fields.

IMPORTANT:
The full N-BaIoT dataset is large. The project uses configurable sampling so a normal PC does not have to load the entire dataset.

## 4. Run the complete project

After placing the dataset in data/raw:

.\run_project.ps1

Or:

python cn_project.py

Then:

python evaluate.py
python make_graphs.py

## 5. Synthetic test mode

Before downloading the real dataset, you can verify the complete pipeline with:

DATA_MODE = "synthetic"

in config.py.

This creates a 10,000-row demonstration dataset following the supplied implementation's feature/label idea.

Synthetic results MUST NOT be presented as real N-BaIoT experimental results.

## 6. Outputs

models/
- original_model.keras
- original_model.h5
- fp32_model.tflite
- quantized_model_int8.tflite

results/
- project_results_summary.csv
- detailed_metrics.csv
- predictions.csv
- classification_report.txt

plots/
- training_history.png
- model_accuracy_comparison.png
- model_size_comparison.png
- inference_time_comparison.png
- confusion_matrix_fp32.png
- confusion_matrix_int8.png

## 7. Main comparison

The generated project_results_summary.csv contains:

- Original/FP32 accuracy
- INT8 accuracy
- FP32 model size
- INT8 model size
- size reduction percentage
- FP32 inference time
- INT8 inference time
- inference speed improvement

Use these measured values in the final report. Do not manually invent values.

## 8. Methodology

Raw network traffic
→ feature selection
→ missing/invalid value handling
→ train/test split
→ StandardScaler fitted only on training data
→ Conv1D CNN
→ FP32 evaluation
→ TFLite FP32
→ representative dataset calibration
→ full INT8 quantization
→ INT8 evaluation
→ size/latency comparison

## 9. Important technical detail

Full integer quantization uses:

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

The evaluator correctly quantizes each input using the TFLite interpreter's input scale and zero-point before invoking the INT8 model.

## 10. Report claim

The correct conclusion must be based on your measured output. Do not assume that INT8 will always improve latency on every CPU.

A safe conclusion format is:

"Post-training INT8 quantization reduced the model storage requirement by X% and changed average inference latency from Y ms to Z ms per sample, while test accuracy changed from A to B."

Replace X, Y, Z, A and B with generated measurements.
