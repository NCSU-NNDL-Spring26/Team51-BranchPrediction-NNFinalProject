"""
branch_predictor_neural.py
Neural Sequence Modeling for Branch Prediction — Production Version
ECE 542 — Phase 2

Aims for <5% misprediction using:
  • PC token embeddings (each unique branch address → learned vector)
  • Long branch history (128 branches)
  • Deep LSTM with regularization
  • Cosine LR schedule + early stopping

Usage:
    python branch_predictor_neural.py
    python branch_predictor_neural.py path/to/trace.txt 1500000

Edit TRACE_FILE below or pass it on the command line.
"""

import sys, time
import numpy as np
import torch
import torch.nn as nn


def accuracy_score(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    return float(np.mean(labels == preds))


def confusion_matrix(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    cm = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(labels, preds):
        cm[int(actual), int(predicted)] += 1
    return cm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRACE_FILE   = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/prajn/Documents/nnn/Projectfinal/gcc_trace.txt"
N_SAMPLES    = int(sys.argv[2]) if len(sys.argv) > 2 else 1_500_000
HISTORY_LEN  = 128
BATCH_SIZE   = 2048
EPOCHS       = 12
LR           = 2e-3
EMBED_DIM    = 32
HIDDEN_SIZE  = 192
NUM_LAYERS   = 2
WEIGHT_DECAY = 1e-5
PATIENCE     = 3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device     : {DEVICE}")
print(f"Trace file : {TRACE_FILE}")
print(f"Samples    : {N_SAMPLES:,}\n")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_trace(path, n):
    pcs, outs = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n: break
            a, b = line.strip().split()
            pcs.append(int(a, 16))
            outs.append(1 if b == 't' else 0)
    return np.array(pcs, dtype=np.int64), np.array(outs, dtype=np.int64)


print("Loading trace …")
t0 = time.time()
pcs_raw, outcomes = load_trace(TRACE_FILE, N_SAMPLES)
print(f"  {len(pcs_raw):,} branches loaded in {time.time()-t0:.1f}s")
print(f"  Taken rate : {outcomes.mean()*100:.2f}%")

unique_pcs, pc_ids = np.unique(pcs_raw, return_inverse=True)
n_pcs = len(unique_pcs)
print(f"  Unique PCs : {n_pcs:,}\n")


# ─────────────────────────────────────────────
# 2. WINDOWS (vectorized)
# ─────────────────────────────────────────────
print("Building windows …")
N2  = len(outcomes) - HISTORY_LEN
idx = np.arange(N2)[:, None] + np.arange(HISTORY_LEN)
Xpc = pc_ids[idx]
Xo  = outcomes[idx].astype(np.float32)
y   = outcomes[HISTORY_LEN:HISTORY_LEN + N2]
print(f"  {N2:,} windows of length {HISTORY_LEN}")

split = int(0.8 * N2)
Xpc_tr, Xpc_te = Xpc[:split], Xpc[split:]
Xo_tr,  Xo_te  = Xo[:split],  Xo[split:]
y_tr,   y_te   = y[:split],   y[split:]
print(f"  Train : {len(y_tr):,}   Test : {len(y_te):,}\n")

Xpc_tr = torch.from_numpy(Xpc_tr); Xo_tr = torch.from_numpy(Xo_tr); y_tr = torch.from_numpy(y_tr)
Xpc_te = torch.from_numpy(Xpc_te); Xo_te = torch.from_numpy(Xo_te); y_te = torch.from_numpy(y_te)


# ─────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────
class MLPPredictor(nn.Module):
    def __init__(self, n_pcs, embed_dim, history_len, hidden=512):
        super().__init__()
        self.pc_emb = nn.Embedding(n_pcs, embed_dim)
        in_features = history_len * (embed_dim + 1)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),  nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )
    def forward(self, xp, xo):
        e = self.pc_emb(xp)
        x = torch.cat([e, xo.unsqueeze(-1)], -1).flatten(1)
        return self.net(x)


class RNNPredictor(nn.Module):
    def __init__(self, n_pcs, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.pc_emb = nn.Embedding(n_pcs, embed_dim)
        self.rnn = nn.RNN(embed_dim + 1, hidden_size,
                          num_layers=num_layers, batch_first=True,
                          dropout=0.2, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, xp, xo):
        e = self.pc_emb(xp)
        x = torch.cat([e, xo.unsqueeze(-1)], -1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMPredictor(nn.Module):
    def __init__(self, n_pcs, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.pc_emb = nn.Embedding(n_pcs, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 1, hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )
    def forward(self, xp, xo):
        e = self.pc_emb(xp)
        x = torch.cat([e, xo.unsqueeze(-1)], -1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ─────────────────────────────────────────────
# 4. TRAIN / EVAL
# ─────────────────────────────────────────────
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_one_epoch(model, criterion, opt):
    model.train()
    n = len(y_tr); perm = torch.randperm(n)
    tot, nb = 0.0, 0
    for s in range(0, n, BATCH_SIZE):
        b = perm[s:s + BATCH_SIZE]
        xp = Xpc_tr[b].to(DEVICE)
        xo = Xo_tr[b].to(DEVICE)
        yy = y_tr[b].to(DEVICE)
        opt.zero_grad()
        loss = criterion(model(xp, xo), yy)
        loss.backward()
        opt.step()
        tot += loss.item(); nb += 1
    return tot / nb


@torch.no_grad()
def evaluate(model):
    model.eval()
    preds = []
    n = len(y_te)
    for s in range(0, n, BATCH_SIZE):
        xp = Xpc_te[s:s+BATCH_SIZE].to(DEVICE)
        xo = Xo_te[s:s+BATCH_SIZE].to(DEVICE)
        preds.append(model(xp, xo).argmax(1).cpu())
    P = torch.cat(preds).numpy()
    L = y_te.numpy()
    return L, P


def run(name, model):
    print(f"\n{'='*65}\n  {name}\n{'='*65}")
    model.to(DEVICE)
    print(f"  Parameters : {count_params(model):,}")
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    t0 = time.time()
    best_acc, best_cm, stale = 0.0, None, 0
    for ep in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, crit, opt)
        sch.step()
        L, P = evaluate(model)
        acc = accuracy_score(L, P); mis = (1 - acc) * 100
        if acc > best_acc:
            best_acc = acc; best_cm = confusion_matrix(L, P); stale = 0
        else:
            stale += 1
        print(f"  Epoch {ep:2d}/{EPOCHS}  loss={loss:.4f}  acc={acc*100:.2f}%  mis={mis:.2f}%  best_mis={(1-best_acc)*100:.2f}%")
        if stale >= PATIENCE:
            print(f"  [early stop — no improvement for {PATIENCE} epochs]")
            break

    train_t = time.time() - t0
    cm = best_cm
    print(f"\n  BEST accuracy   : {best_acc*100:.2f}%")
    print(f"  BEST misrate    : {(1 - best_acc)*100:.2f}%")
    print(f"  Train time      : {train_t:.1f}s")
    print(f"  Confusion matrix (best epoch):")
    print(f"    TN={cm[0,0]:,}   FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}   TP={cm[1,1]:,}")
    return {"name": name, "params": count_params(model),
            "acc": best_acc * 100, "mis": (1 - best_acc) * 100, "time": train_t}


# ─────────────────────────────────────────────
# 5. RUN ALL
# ─────────────────────────────────────────────
results = []
results.append(run("MLP",  MLPPredictor (n_pcs, EMBED_DIM, HISTORY_LEN, hidden=512)))
results.append(run("RNN",  RNNPredictor (n_pcs, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS)))
results.append(run("LSTM", LSTMPredictor(n_pcs, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS)))


# ─────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*70}\n  FINAL RESULTS\n{'='*70}")
print(f"  {'Model':<8} {'Acc':>9} {'Misrate':>10} {'Params':>14} {'Time':>8}")
print(f"  {'-'*8} {'-'*9} {'-'*10} {'-'*14} {'-'*8}")
baselines = [
    ("Trivial",  51.48, 48.52, 0,       0),
    ("LogReg",   57.80, 42.20, 65,      24.2),
    ("DTree",    65.60, 34.40, 5119,    20.9),
    ("RForest",  67.85, 32.15, 666102,  55.7),
]
for n, a, m, p, t in baselines:
    print(f"  {n:<8} {a:>8.2f}% {m:>9.2f}% {p:>14,} {t:>7.1f}s")
print(f"  {'─'*55}")
for r in results:
    print(f"  {r['name']:<8} {r['acc']:>8.2f}% {r['mis']:>9.2f}% "
          f"{r['params']:>14,} {r['time']:>7.1f}s")
print()
