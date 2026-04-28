"""
Baseline Branch Predictors — Classical ML Models
Project: Neural Sequence Modeling for Microarchitectural Branch Prediction

Models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest

Metrics:
  - Misprediction Rate (primary)
  - Accuracy
  - Parameter Footprint (hardware budget proxy)
"""

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.metrics      import accuracy_score, confusion_matrix, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
TRACE_FILE  = "C:\\Users\\shreeya\\OneDrive\\Desktop\\Masters\\NN_(542)\\Project Branch Prediction\\gcc_trace.txt"
HISTORY_LEN = 32        # sliding window: how many past branches = features
MAX_ROWS    = 500_000   # use first 500k rows (file has 2M — keeps runtime fast)
TRAIN_RATIO = 0.80      # 80% train, 20% test — temporal split, no shuffling

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD TRACE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING TRACE FILE")
print("=" * 60)

pcs, outcomes = [], []
with open(TRACE_FILE) as f:
    for i, line in enumerate(f):
        if i >= MAX_ROWS:
            break
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        pc_hex, outcome = parts
        pcs.append(int(pc_hex, 16))
        outcomes.append(1 if outcome == 't' else 0)

pcs      = np.array(pcs,      dtype=np.int64)
outcomes = np.array(outcomes, dtype=np.int8)

taken_rate     = outcomes.mean() * 100
not_taken_rate = 100 - taken_rate

print(f"  Total branches loaded : {len(outcomes):,}")
print(f"  Taken     (label=1)   : {taken_rate:.1f}%")
print(f"  Not-Taken (label=0)   : {not_taken_rate:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD SLIDING-WINDOW FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# For each branch i, features = [last 32 normalized PCs] + [last 32 outcomes]
# Total features per sample = HISTORY_LEN * 2 = 64
print(f"\n{'=' * 60}")
print(f"BUILDING FEATURES  (history window = {HISTORY_LEN})")
print(f"{'=' * 60}")

pc_min  = pcs.min()
pc_max  = pcs.max()
pc_norm = (pcs - pc_min) / (pc_max - pc_min + 1e-9)   # normalize PCs to [0,1]

X, y = [], []
for i in range(HISTORY_LEN, len(outcomes)):
    window_pcs      = pc_norm[i - HISTORY_LEN : i]
    window_outcomes = outcomes[i - HISTORY_LEN : i].astype(np.float32)
    X.append(np.concatenate([window_pcs, window_outcomes]))
    y.append(outcomes[i])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int8)

print(f"  Feature matrix shape : {X.shape}")
print(f"  Features per sample  : {X.shape[1]}  (32 PCs + 32 outcomes)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPORAL TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
split   = int(len(y) * TRAIN_RATIO)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n  Train samples : {len(y_train):,}")
print(f"  Test  samples : {len(y_test):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRIVIAL BASELINE — always predict majority class
# ─────────────────────────────────────────────────────────────────────────────
majority      = int(np.round(y_train.mean()))
trivial_preds = np.full(len(y_test), majority)
trivial_acc   = accuracy_score(y_test, trivial_preds)
trivial_misp  = (1 - trivial_acc) * 100
label_name    = "Taken" if majority else "Not-Taken"

print(f"\n{'=' * 60}")
print(f"TRIVIAL BASELINE  (always predict {label_name})")
print(f"{'=' * 60}")
print(f"  Accuracy           : {trivial_acc*100:.2f}%")
print(f"  Misprediction Rate : {trivial_misp:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=300,
        solver="saga",
        C=1.0,
        n_jobs=-1,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        n_jobs=-1,
        random_state=42
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN + EVALUATE EACH MODEL
# ─────────────────────────────────────────────────────────────────────────────
results = {}

for name, model in models.items():
    print(f"\n{'=' * 60}")
    print(f"MODEL: {name}")
    print(f"{'=' * 60}")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    misp = (1 - acc) * 100
    cm   = confusion_matrix(y_test, y_pred)

    # Approximate parameter / node count per model type
    if hasattr(model, 'coef_'):
        n_params = model.coef_.size + model.intercept_.size
    elif hasattr(model, 'estimators_'):
        n_params = sum(t.tree_.node_count for t in model.estimators_)
    else:
        n_params = model.tree_.node_count

    results[name] = {
        "accuracy"  : acc * 100,
        "misp_rate" : misp,
        "train_time": train_time,
        "n_params"  : n_params,
        "confusion" : cm,
    }

    print(f"  Training time      : {train_time:.1f}s")
    print(f"  Accuracy           : {acc*100:.2f}%")
    print(f"  Misprediction Rate : {misp:.2f}%")
    print(f"  Parameter Footprint: {n_params:,}")
    print(f"\n  Confusion Matrix:")
    print(f"                       Pred Not-Taken        Pred Taken")
    print(f"  True Not-Taken  : {cm[0,0]:>13,}(TN)   {cm[0,1]:>10,}(FP)")
    print(f"  True Taken      : {cm[1,0]:>13,}(FN)   {cm[1,1]:>10,}(TP)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not-Taken", "Taken"]))

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("SUMMARY COMPARISON")
print(f"{'=' * 60}")
header = f"{'Model':<22} {'Accuracy':>10} {'Misp Rate':>10} {'# Params':>12} {'Train Time':>12}"
print(header)
print("-" * 70)
print(f"{'Trivial Baseline':<22} {trivial_acc*100:>9.2f}% {trivial_misp:>9.2f}%  {'N/A':>11}  {'N/A':>10}")
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:>9.2f}% {r['misp_rate']:>9.2f}%  {r['n_params']:>11,}  {r['train_time']:>9.1f}s")
print(f"{'=' * 60}")
