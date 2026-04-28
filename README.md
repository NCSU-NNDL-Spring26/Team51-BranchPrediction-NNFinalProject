# Neural Sequence Modeling for Microarchitectural Branch Prediction

> Treating CPU instruction execution as a sequence modeling problem using classical ML and deep neural networks to predict branch outcomes in real processor traces.

---

## 👥 Team
- Shreeya Ranwadkar
- Advika Metre
- Prajna Hassan Umesh

---

## 🗂️ Project Structure

```
Team51-BranchPrediction-NNFinalProject/
│
├── Code_Files/
│   ├── ML_models.py                   ← Classical ML models (Phase 1)
│   └── branch_predictor_neural.py     ← Neural Network models (Phase 2)
│
├── Dataset/
│   ├── gcc_trace.txt                  ← GCC workload trace (2M branch records)
│   └── perl_trace.txt                 ← Perl workload trace (2M branch records)
│
├── Screenshots/
│   ├── ML_Results/                    ← Baseline ML model result screenshots
│   └── NN_Results/                    ← Neural network model result screenshots
│
├── NNProject.pptx                     ← Project slide deck
├── NN_PF2_Team51.mp4                  ← Video presentation
└── README.md
```

---

## 📁 File Descriptions

| File | Description |
|---|---|
| `Code_Files/ML_models.py` | Classical ML baseline models — Logistic Regression, Decision Tree, Random Forest |
| `Code_Files/branch_predictor_neural.py` | Neural network models — MLP, RNN, LSTM using PyTorch |
| `Dataset/gcc_trace.txt` | GCC workload instruction trace dataset |
| `Dataset/perl_trace.txt` | Perl workload instruction trace dataset |
| `NNProject.pptx` | Project slide deck |
| `NN_PF2_Team51.mp4` | Video presentation |

---

## 📊 Dataset

- **Source:** Championship Branch Prediction (CBP) Framework
- **Workloads:** GCC and Perl compiler benchmarks (SPEC CPU)
- **Size:** ~2,000,000 consecutive branch records per trace
- **Format:** Each row contains:
  - `PC address` — Hexadecimal program counter identifying the branch instruction
  - `outcome` — `t` (Taken = 1) or `n` (Not-Taken = 0)

**Example:**
```
302d28 n
305b0c t
30093c t
```

**Class Distribution (500k sample — gcc_trace):**
- Taken (1): 57.3%
- Not-Taken (0): 42.7%

---

## 🔧 Feature Engineering

For each branch `i`, a sliding window of the previous 32 branches is used as input:

- **MLP input:** Flat vector of 64 features `[32 normalized PCs + 32 past outcomes]`
- **RNN/LSTM input:** Sequence of shape `(32, 2)` — each timestep is `[PC, outcome]`
- **Train/Test Split:** 80% train / 20% test — temporal split (no shuffling) to preserve sequence order

---

## 🤖 Models

### Phase 1 — Classical ML Baselines (`Code_Files/ML_models.py`)

| Model | Description |
|---|---|
| **Trivial Baseline** | Always predicts majority class (Taken) — zero learning, sanity check floor |
| **Logistic Regression** | Linear classifier, one weight per feature |
| **Decision Tree** | Tree of if/else decision rules, max depth 15 |
| **Random Forest** | Ensemble of 100 decision trees, max depth 15 |

### Phase 2 — Neural Networks (`Code_Files/branch_predictor_neural.py`)

| Model | Description |
|---|---|
| **MLP** | Multi-Layer Perceptron — feedforward network with hidden layers (128 → 64 → 32 → 1) |
| **RNN** | Vanilla Recurrent Neural Network — processes branch history as a sequence with memory |
| **LSTM** | Long Short-Term Memory — advanced RNN with forget gate, captures long-range dependencies across thousands of branches |

---

## 📈 Results

### Classical ML Models
ML Results

### Neural Network Models
NN Results

---

## 📏 Evaluation Metrics

| Metric | Description |
|---|---|
| **Misprediction Rate** | Primary metric — % of wrong predictions. Every misprediction in a real CPU flushes the pipeline (~15 wasted cycles) |
| **Accuracy** | % of correct predictions overall |
| **Parameter Footprint** | Total learned parameters — proxy for hardware silicon area and memory cost |
| **Confusion Matrix** | TN / TP / FP / FN breakdown showing where the model fails |
| **Precision** | Of all "Taken" predictions, how many were actually Taken |
| **Recall** | Of all actual Taken branches, how many did the model catch |
| **F1-Score** | Harmonic mean of Precision and Recall |

---

## ⚙️ Setup & Installation

### Requirements
```bash
pip install torch numpy scikit-learn
```

### Running the Baseline Models
```bash
python Code_Files/ML_models.py
```

### Running the Neural Network Models
```bash
python Code_Files/branch_predictor_neural.py
```

> ⚠️ Make sure to update the trace file path at the top of each script:
> ```python
> TRACE_FILE = "Dataset/gcc_trace.txt"
> ```

---

## 🔑 Key Findings

- **Random Forest** is the best classical ML model (32.15% misprediction) but costs 666K parameters — too expensive for real hardware
- **Logistic Regression** has only 65 parameters but weak accuracy (42.20% misprediction) — too simple
- **LSTM** achieves the best accuracy with far fewer parameters than Random Forest — best candidate for hardware deployment
- Branch prediction is inherently hard — even state-of-the-art hardware predictors (TAGE) achieve only ~5–10% misprediction after decades of research

---

*Course Project — Neural Sequence Modeling for Advanced Microarchitectural Branch Prediction*
