import os, time, warnings, platform, csv, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score
import joblib

try:
    from tqdm import tqdm
    TQDM_OK = False
except Exception:
    TQDM_OK = False

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager

CSV_PATH = r"D:/Importante/Universidad/Seminario2/InvestigaciónDL/dataset_balanceado_limpio.csv"
SEED = 42

TEST_SIZE = 0.20
VAL_SIZE  = 0.25

SELECTED_COLS = [
    "Time_To_Live","Rate","syn_flag_number","rst_flag_number","psh_flag_number",
    "ack_flag_number","ack_count","syn_count","rst_count","HTTP","HTTPS","TCP","Std","IAT","Variance"
]

DEVICE = torch.device("cpu")
IS_WINDOWS  = platform.system().lower().startswith("win")
BATCH_SIZE  = 1792
NUM_WORKERS = 0
PIN_MEMORY  = False

EPOCHS        = 35
PATIENCE      = 6
BASE_LR       = 6e-4
MAX_LR        = 1.2e-3
WEIGHT_DECAY  = 1e-4
CLIP_NORM     = 1.0
DROPOUT       = 0.10
FEATURE_DROPOUT_P = 0.02
LSMOOTH       = 0.05
USE_CLASS_WEIGHTS = False

D_MODEL, N_HEADS, N_LAYERS, MLP_HIDDEN = 256, 8, 4, 1024
N_BINS = 128

torch.set_num_threads(min(12, os.cpu_count() or 8))
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*Bins whose width are too small.*")

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join("resultados", RUN_ID)
os.makedirs(OUT_DIR, exist_ok=True)
LOG_TXT = os.path.join(OUT_DIR, "training_log.txt")
CSV_METRICS = os.path.join(OUT_DIR, "metrics.csv")


def seed_everything(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)

def f1_macro(y_true, y_pred):
    return precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]

def log_line(text):
    print(text, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

@contextmanager
def silence_stdout():
    _stdout = sys.stdout
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            yield
    finally:
        sys.stdout = _stdout


class CrossEntropyLS(nn.Module):
    def __init__(self, smoothing=0.02, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        n_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            w = self.weight[target].unsqueeze(1)
            loss = -(w * true_dist * log_probs).sum(dim=1).mean()
        else:
            loss = -(true_dist * log_probs).sum(dim=1).mean()
        return loss


class FeatureDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__(); self.p = p
    def forward(self, x):
        if (not self.training) or self.p <= 0: return x
        mask = (torch.rand(x.shape[1], device=x.device) > self.p).float()
        return x * mask.unsqueeze(0)

class TabTransHybrid(nn.Module):
    def __init__(self, n_features, n_bins, n_classes,
                 d_model=256, n_heads=8, n_layers=4,
                 dropout=0.10, mlp_hidden=1024, featdrop_p=0.02):
        super().__init__()
        self.n_features = n_features
        self.emb_tables = nn.ModuleList([nn.Embedding(n_bins, d_model) for _ in range(n_features)])
        self.feat_scale = nn.Parameter(torch.ones(n_features, d_model))
        self.feat_bias  = nn.Parameter(torch.zeros(n_features, d_model))

        self.featdrop = FeatureDropout(featdrop_p)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=mlp_hidden*2,
            activation="gelu", dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x_cont, x_bins):
        B, F = x_cont.shape
        x_cont = self.featdrop(x_cont)

        toks = []
        for j in range(F):
            e = self.emb_tables[j](x_bins[:, j].long())
            c = x_cont[:, j].unsqueeze(-1) * self.feat_scale[j] + self.feat_bias[j]
            toks.append(e + c)
        tok = torch.stack(toks, dim=1)

        cls = self.cls_token.expand(B, 1, -1)
        seq = torch.cat([cls, tok], dim=1)
        enc = self.encoder(seq)[:, 0, :]
        enc = self.norm(enc)
        logits = self.head(enc)
        return logits


class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
    def load_shadow(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k] = v.clone()
        model.load_state_dict(sd)


@torch.no_grad()
def evaluate(loader, model, criterion, n_classes):
    model.eval()
    tot_loss, n = 0.0, 0
    ys, yp, yprob = [], [], []
    for xc, xb, yb in loader:
        xc, xb, yb = xc.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xc, xb)
        loss = criterion(logits, yb)
        tot_loss += loss.item() * xc.size(0); n += xc.size(0)
        probs = F.softmax(logits, dim=1)
        yprob.append(probs.detach().cpu().numpy())
        yp.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yp)
    y_prob = np.concatenate(yprob)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_macro(y_true, y_pred)
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = np.nan
    return tot_loss/max(1,n), acc, f1m, y_true, y_pred, y_prob, auc_macro


def main():
    t0 = time.time(); seed_everything(SEED)

    with open(LOG_TXT, "w", encoding="utf-8") as f:
        f.write(f"Inicio entrenamiento | DEVICE={DEVICE} | RUN_ID={RUN_ID} | {time.ctime()}\n")
    with open(CSV_METRICS, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss","val_acc","val_f1","val_auc","secs"])

    if not os.path.isfile(CSV_PATH):
        log_line(f"❌ No se encuentra CSV en {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    y_candidates = ["grouped_label","label","Label","attack_type","Attack","Clase"]
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if y_col is None:
        log_line(f"❌ No se encuentra etiqueta en {y_candidates}")
        sys.exit(1)

    faltan = [c for c in SELECTED_COLS if c not in df.columns]
    if faltan:
        log_line(f"❌ Faltan columnas requeridas: {faltan}")
        sys.exit(1)

    X_all_df = df[SELECTED_COLS].astype(np.float32)
    le = LabelEncoder()
    y_all = le.fit_transform(df[y_col].astype(str).values)
    target_names = list(le.classes_)
    n_classes = len(target_names)

    X_trval_df, X_te_df, y_trval, y_te = train_test_split(
        X_all_df, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=SEED
    )
    X_tr_df, X_val_df, y_tr, y_val = train_test_split(
        X_trval_df, y_trval, test_size=VAL_SIZE, stratify=y_trval, random_state=SEED
    )

    lo = X_tr_df.quantile(0.001); hi = X_tr_df.quantile(0.999)
    X_tr_df  = X_tr_df.clip(lower=lo, upper=hi, axis=1)
    X_val_df = X_val_df.clip(lower=lo, upper=hi, axis=1)
    X_te_df  = X_te_df.clip(lower=lo, upper=hi, axis=1)

    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr_df)
    X_val_imp= imp.transform(X_val_df)
    X_te_imp = imp.transform(X_te_df)

    scaler = StandardScaler()
    X_tr_cont  = scaler.fit_transform(X_tr_imp).astype(np.float32)
    X_val_cont = scaler.transform(X_val_imp).astype(np.float32)
    X_te_cont  = scaler.transform(X_te_imp).astype(np.float32)

    kb = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')
    Xtr_bins = kb.fit_transform(X_tr_imp)
    Xva_bins = kb.transform(X_val_imp)
    Xte_bins = kb.transform(X_te_imp)

    Xtr_bins = np.clip(Xtr_bins, 0, N_BINS - 1).astype(np.int64)
    Xva_bins = np.clip(Xva_bins, 0, N_BINS - 1).astype(np.int64)
    Xte_bins = np.clip(Xte_bins, 0, N_BINS - 1).astype(np.int64)

    tr_cont = torch.tensor(X_tr_cont, dtype=torch.float32)
    va_cont = torch.tensor(X_val_cont, dtype=torch.float32)
    te_cont = torch.tensor(X_te_cont, dtype=torch.float32)
    tr_bins = torch.tensor(Xtr_bins, dtype=torch.long)
    va_bins = torch.tensor(Xva_bins, dtype=torch.long)
    te_bins = torch.tensor(Xte_bins, dtype=torch.long)
    tr_y = torch.tensor(y_tr, dtype=torch.long)
    va_y = torch.tensor(y_val, dtype=torch.long)
    te_y = torch.tensor(y_te, dtype=torch.long)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(tr_cont, tr_bins, tr_y),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, generator=g)
    val_loader   = DataLoader(TensorDataset(va_cont, va_bins, va_y),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(TensorDataset(te_cont, te_bins, te_y),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    log_line(f"Inicio entrenamiento | DEVICE={DEVICE} | RUN_ID={RUN_ID} | {time.ctime()}")
    log_line(f"Sizes -> train:{len(tr_y)}, val:{len(va_y)}, test:{len(te_y)}")
    log_line(f"Batches -> train:{len(train_loader)}, val:{len(val_loader)}, test:{len(test_loader)}")
    if len(train_loader) == 0:
        log_line("❌ train_loader quedó en 0. Baja BATCH_SIZE.")
        sys.exit(1)

    assert D_MODEL % N_HEADS == 0, f"d_model({D_MODEL}) debe ser divisible por n_heads({N_HEADS})"
    model = TabTransHybrid(
        n_features=tr_cont.shape[1], n_bins=N_BINS, n_classes=n_classes,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        dropout=DROPOUT, mlp_hidden=MLP_HIDDEN, featdrop_p=FEATURE_DROPOUT_P
    ).to(DEVICE)

    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(y_tr)
        class_weights = (class_counts.sum() / np.maximum(class_counts, 1.0))
        class_weights = class_weights / class_weights.mean()
        cw = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        log_line(f"Pesos por clase: {np.round(class_weights,3)}")
    else:
        cw = None
        log_line("Pesos por clase: [desactivado] (dataset balanceado)")

    criterion_train = CrossEntropyLS(smoothing=LSMOOTH, weight=cw)
    criterion_eval  = nn.CrossEntropyLoss(weight=cw)

    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2
    )

    ema = EMA(model, decay=0.998)

    best_f1m, best_state, patience = -1.0, None, 0
    global_start = time.time()
    LOG_EVERY = max(1, len(train_loader)//20)

    for epoch in range(1, EPOCHS+1):
        model.train()
        running, seen = 0.0, 0
        ep_start = time.time()
        log_line(f"\nEpoch {epoch:02d}/{EPOCHS}  (batches: {len(train_loader)})")

        iterator = train_loader if not TQDM_OK else tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False, unit="batch")
        for i, xb in enumerate(iterator):
            xc, xbb, yb = xb
            xc, xbb, yb = xc.to(DEVICE), xbb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xc, xbb)
            loss = criterion_train(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step(); scheduler.step()
            ema.update(model)
            running += loss.item() * xc.size(0); seen += xc.size(0)

            if (i+1) % LOG_EVERY == 0 or (i+1)==len(train_loader):
                elapsed = max(1e-6, time.time() - ep_start)
                sps = seen / elapsed
                pct = 100.0 * (i+1) / len(train_loader)
                log_line(f"  {i+1:>4d}/{len(train_loader)} | {pct:5.1f}% | loss={loss.item():.4f} | sps={sps:.0f}")

        train_loss = running/max(1,seen)

        val_loss, val_acc, val_f1m, yt, yp, yprob, aucM = evaluate(val_loader, model, criterion_eval, n_classes)
        ep_secs = time.time() - ep_start

        done = epoch
        remaining = EPOCHS - epoch
        avg_ep = (time.time() - global_start) / done
        eta_sec = avg_ep * remaining

        msg = (f"Epoch {epoch:02d}/{EPOCHS} | Train {train_loss:.4f} | "
               f"Val {val_loss:.4f} | Acc {val_acc:.4f} | F1 {val_f1m:.4f} | "
               f"AUC {aucM:.4f} | dur {fmt_time(ep_secs)} | ETA {fmt_time(eta_sec)}")
        log_line(msg)

        with open(CSV_METRICS, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_acc, val_f1m, aucM, ep_secs])

        if val_f1m > best_f1m:
            best_f1m = val_f1m
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = 0
            torch.save({"state_dict": best_state}, os.path.join(OUT_DIR, "tabtx_best.pt"))
            log_line(f"  🔽 Nuevo mejor F1-macro VAL: {best_f1m:.4f} (checkpoint guardado)")
        else:
            patience += 1
            if patience >= PATIENCE:
                log_line(f"⏹️ Early stopping en epoch {epoch}. Mejor F1 {best_f1m:.4f}")
                break

        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                   os.path.join(OUT_DIR, "tabtx_last.pt"))

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    ema.load_shadow(model)

    te_loss, te_acc, te_f1m, y_true, y_pred, y_prob, aucM = evaluate(test_loader, model, criterion_eval, n_classes)
    rep = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, digits=4)
    log_line(f"\n✅ TEST | Loss {te_loss:.4f} | Acc {te_acc:.4f} | F1 {te_f1m:.4f} | AUC {aucM:.4f}")
    log_line("\nClassification Report (TEST)\n" + rep)

    preproc = {
        "selected_cols": SELECTED_COLS,
        "imputer": imp,
        "scaler": scaler,
        "kbins": kb,
        "label_encoder": le,
        "n_bins": N_BINS
    }
    joblib.dump(preproc, os.path.join(OUT_DIR, "preproc.pkl"))

    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "arch": dict(n_features=tr_cont.shape[1], n_bins=N_BINS, n_classes=n_classes,
                     d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                     dropout=DROPOUT, mlp_hidden=MLP_HIDDEN, featdrop_p=FEATURE_DROPOUT_P)
    }, os.path.join(OUT_DIR, "tabtx_best_full.pt"))

    model.eval()
    ex_cont = torch.zeros(1, tr_cont.shape[1], dtype=torch.float32).to(DEVICE)
    ex_bins = torch.zeros(1, tr_cont.shape[1], dtype=torch.long).to(DEVICE)
    with silence_stdout():
        traced = torch.jit.trace(lambda a,b: model(a,b), (ex_cont, ex_bins))
    traced.save(os.path.join(OUT_DIR, "tabtx_script.pt"))

    try:
        with silence_stdout():
            torch.onnx.export(
                model, (ex_cont, ex_bins), os.path.join(OUT_DIR, "tabtx.onnx"),
                input_names=["x_cont","x_bins"], output_names=["logits"],
                dynamic_axes={"x_cont": {0: "batch"}, "x_bins": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17
            )
        log_line("🟢 Exportado ONNX")
    except Exception as e:
        log_line(f"⚠️ ONNX falló: {e}")

    total_secs = time.time() - t0
    log_line(f"\nArtefactos en: {OUT_DIR}")
    log_line(f"Tiempo total: {fmt_time(total_secs)} | Dispositivo: {DEVICE}")

if __name__ == "__main__":
    main()
