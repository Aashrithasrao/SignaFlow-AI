import os
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "dataset/processed"
MODEL_DIR = "models"

X_PATH = f"{DATA_DIR}/X.npy"
Y_PATH = f"{DATA_DIR}/y.npy"
CLASSES_PATH = f"{DATA_DIR}/classes.npy"

MODEL_V3_PATH = f"{MODEL_DIR}/sign_model_50_v3.keras"
MODEL_HYBRID_PATH = f"{MODEL_DIR}/sign_model_50_hybrid_cnn.keras"
OUTPUT_REPORT_PATH = f"{MODEL_DIR}/model_comparison_results.json"

print("=" * 75)
print("SIGNAFLOW AI — DIRECT MODEL COMPARISON & SENTENCE RECOGNITION BENCHMARK")
print("=" * 75)

# Load data
X = np.load(X_PATH)
y = np.load(Y_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

_, X_val, _, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Validation Samples: {len(X_val):,}")
print(f"Total Sign Classes: {len(classes)}")

# Load models
print(f"\nLoading Model 1 (BiLSTM V3): {MODEL_V3_PATH}")
model_v3 = tf.keras.models.load_model(MODEL_V3_PATH)

print(f"Loading Model 2 (Hybrid 1D-CNN + BiLSTM): {MODEL_HYBRID_PATH}")
model_hybrid = tf.keras.models.load_model(MODEL_HYBRID_PATH)


def evaluate_model_pipeline(model, name, X_data, y_data):
    # Latency test (100 runs)
    dummy_input = X_data[:1]
    # Warmup
    for _ in range(10):
        _ = model(dummy_input, training=False)
    
    start_time = time.time()
    n_latency_runs = 100
    for i in range(n_latency_runs):
        _ = model(X_data[i:i+1], training=False)
    avg_latency_ms = ((time.time() - start_time) / n_latency_runs) * 1000

    # Predictions
    loss, acc = model.evaluate(X_data, y_data, verbose=0)
    probabilities = model.predict(X_data, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_data, predictions, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_data, predictions, average="macro", zero_division=0
    )

    # Confidence and stability metrics
    top1_conf = np.max(probabilities, axis=1)
    
    # Margin between top 1 and top 2 predictions
    sorted_probs = np.sort(probabilities, axis=1)
    top2_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Percentage of predictions that pass sentence stability gates
    pct_over_45 = np.mean(top1_conf >= 0.45) * 100
    pct_over_60 = np.mean(top1_conf >= 0.60) * 100
    pct_stable_margin = np.mean(top2_margin >= 0.15) * 100

    # Per-class scores
    per_class_f1 = {}
    for idx, cname in enumerate(classes):
        mask = (y_data == idx)
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == idx)
            per_class_f1[str(cname)] = round(float(class_acc) * 100, 2)

    return {
        "model_name": name,
        "loss": round(float(loss), 4),
        "accuracy": round(float(acc) * 100, 2),
        "weighted_precision": round(float(p_w) * 100, 2),
        "weighted_recall": round(float(r_w) * 100, 2),
        "weighted_f1": round(float(f1_w) * 100, 2),
        "macro_f1": round(float(f1_m) * 100, 2),
        "latency_ms": round(float(avg_latency_ms), 2),
        "avg_confidence": round(float(np.mean(top1_conf)) * 100, 2),
        "avg_top2_margin": round(float(np.mean(top2_margin)) * 100, 2),
        "pct_over_45_conf": round(float(pct_over_45), 2),
        "pct_over_60_conf": round(float(pct_over_60), 2),
        "pct_stable_margin": round(float(pct_stable_margin), 2),
        "per_class_accuracy": per_class_f1,
    }


print("\nEvaluating Model 1 (BiLSTM V3)...")
results_v3 = evaluate_model_pipeline(model_v3, "BiLSTM (Model V3)", X_val, y_val)

print("Evaluating Model 2 (Hybrid 1D-CNN + BiLSTM)...")
results_hybrid = evaluate_model_pipeline(model_hybrid, "Hybrid 1D-CNN + BiLSTM", X_val, y_val)

print("\n" + "=" * 75)
print(f"{'METRIC':<35} | {'BiLSTM (Model V3)':<18} | {'Hybrid 1D-CNN + BiLSTM':<22}")
print("-" * 75)
print(f"{'Validation Accuracy':<35} | {results_v3['accuracy']:>16.2f}% | {results_hybrid['accuracy']:>20.2f}%")
print(f"{'Weighted F1-Score':<35} | {results_v3['weighted_f1']:>16.2f}% | {results_hybrid['weighted_f1']:>20.2f}%")
print(f"{'Macro F1-Score':<35} | {results_v3['macro_f1']:>16.2f}% | {results_hybrid['macro_f1']:>20.2f}%")
print(f"{'Validation Loss':<35} | {results_v3['loss']:>17.4f} | {results_hybrid['loss']:>21.4f}")
print(f"{'Inference Latency (per sign)':<35} | {results_v3['latency_ms']:>15.2f}ms | {results_hybrid['latency_ms']:>19.2f}ms")
print(f"{'Average Top-1 Confidence':<35} | {results_v3['avg_confidence']:>16.2f}% | {results_hybrid['avg_confidence']:>20.2f}%")
print(f"{'Average Margin (Top 1 - Top 2)':<35} | {results_v3['avg_top2_margin']:>16.2f}% | {results_hybrid['avg_top2_margin']:>20.2f}%")
print(f"{'Predictions with Conf >= 45%':<35} | {results_v3['pct_over_45_conf']:>16.2f}% | {results_hybrid['pct_over_45_conf']:>20.2f}%")
print(f"{'Predictions with Conf >= 60%':<35} | {results_v3['pct_over_60_conf']:>16.2f}% | {results_hybrid['pct_over_60_conf']:>20.2f}%")
print(f"{'Stable Predictions (Margin >= 15%)':<35} | {results_v3['pct_stable_margin']:>16.2f}% | {results_hybrid['pct_stable_margin']:>20.2f}%")
print("=" * 75)

# Conversational signs sample
common_signs = ["hello", "water", "food", "drink", "yes", "no", "thankyou", "please", "happy", "sad", "mom", "dad"]
print("\n" + "=" * 75)
print(f"{'SAMPLE CONVERSATIONAL SIGN':<35} | {'BiLSTM Acc (%)':<18} | {'Hybrid CNN Acc (%)':<22}")
print("-" * 75)
for s in common_signs:
    v3_score = results_v3["per_class_accuracy"].get(s, "N/A")
    hy_score = results_hybrid["per_class_accuracy"].get(s, "N/A")
    print(f"{s:<35} | {str(v3_score):>18} | {str(hy_score):>22}")
print("=" * 75)

comparison_output = {
    "model_v3": results_v3,
    "model_hybrid": results_hybrid,
}

with open(OUTPUT_REPORT_PATH, "w") as f:
    json.dump(comparison_output, f, indent=2)

print(f"\nComplete comparison report saved to: {OUTPUT_REPORT_PATH}")
