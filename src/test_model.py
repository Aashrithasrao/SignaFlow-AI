import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

X_PATH = "dataset/processed/X.npy"
Y_PATH = "dataset/processed/y.npy"
CLASSES_PATH = "dataset/processed/classes.npy"

MODEL_PATH = "models/sign_model_50_v2.keras"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SIGN LANGUAGE MODEL TEST")
print("=" * 70)

print("\nLoading data...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print(
    f"Samples: {len(X):,}"
)

print(
    f"Classes: {len(classes)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# CREATE SAME VALIDATION SPLIT
# ============================================================

from sklearn.model_selection import train_test_split


_, X_val, _, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


print(
    f"\nValidation samples: "
    f"{len(X_val):,}"
)


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating...")

loss, accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)


print(
    f"\nValidation loss: "
    f"{loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_val,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_val,
    predictions,
    labels=np.arange(len(classes)),
    target_names=[
        str(c)
        for c in classes
    ],
    zero_division=0
)

print(report)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

number_to_show = min(
    30,
    len(X_val)
)

for i in range(
    number_to_show
):

    actual = str(
        classes[y_val[i]]
    )

    predicted = str(
        classes[predictions[i]]
    )

    confidence = float(
        probabilities[i][
            predictions[i]
        ]
    )

    status = (
        "✓"
        if actual == predicted
        else "✗"
    )

    print(
        f"{status} "
        f"Actual: {actual:15s} "
        f"Predicted: {predicted:15s} "
        f"Confidence: {confidence:.2%}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TEST COMPLETE")
print("=" * 70)
