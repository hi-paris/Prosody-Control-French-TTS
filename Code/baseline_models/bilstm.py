import json
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

# ======================
# 1. HYPERPARAMETERS
# ======================
EMBEDDING_MODEL = 'all-mpnet-base-v2'
EMBED_DIM       = 768
SEQ_LEN         = 3   # Default sequence length
HIDDEN_SIZE     = 40
DENSE_SIZE      = 20
OUTPUT_SIZE     = 3   # [pitch, volume, rate]
BATCH_SIZE      = 64
LR              = 1e-3
EPOCHS          = 5
TRAIN_SPLIT     = 0.75  # by episode
VAL_SPLIT       = 0.15  # of train windows
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======================
# 2. DATA PREPARATION
# ======================
def load_and_calibrate(json_file_path):
    """
    Load JSON, extract per-episode texts & raw prosody percentages,
    compute global mean & std for z-score conversion.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ep_texts, ep_raw, all_raw = {}, {}, []
    for ep_id, ep_data in data.items():
        texts, raw_pcts = [], []
        for item in ep_data.get('y', {}).get('parsed_sequence', []):
            if item.get('type') == 'text':
                txt = item.get('text', '').strip()
                if not txt:
                    continue
                texts.append(txt)
                pros = item.get('prosody', {})
                p = float(pros.get('pitch', '+0.00%').strip('%'))
                v = float(pros.get('volume', '+0.00%').strip('%'))
                r = float(pros.get('rate', '+0.00%').strip('%'))
                raw_pcts.append([p, v, r])
                all_raw.append([p, v, r])
        if texts:
            ep_texts[ep_id] = texts
            ep_raw[ep_id]   = np.array(raw_pcts, dtype=np.float32)

    arr = np.vstack(all_raw)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero
    return ep_texts, ep_raw, mean, std


def build_sequences(ep_texts, ep_raw, mean, std, seq_len, eps_filter=None):
    """
    Build sliding windows for episodes in eps_filter (or all if None).
    Returns sequences (N, seq_len, dim), targets_z (N, 3), and raw targets (N, 3).
    """
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    seqs, tgts_z, raw_tgts = [], [], []

    for ep_id, texts in ep_texts.items():
        if eps_filter and ep_id not in eps_filter:
            continue
        embs = embedder.encode(
            texts,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        raw  = ep_raw[ep_id]
        if len(embs) < seq_len:
            continue

        for i in range(len(embs) - seq_len + 1):
            window = embs[i:i + seq_len]
            center = raw[i + seq_len // 2]
            z      = (center - mean) / std
            seqs.append(window)
            tgts_z.append(z)
            raw_tgts.append(center)

    return (
        np.stack(seqs, dtype=np.float32),
        np.stack(tgts_z, dtype=np.float32),
        np.stack(raw_tgts, dtype=np.float32)
    )


# ======================
# 3. DATASET
# ======================
class ProsodyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ======================
# 4. MODEL
# ======================
class BiLSTMProsody(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Linear(HIDDEN_SIZE * 2, DENSE_SIZE)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(DENSE_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = self.norm(x)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.act(self.fc1(h))
        return self.fc2(h)


# ======================
# 5. TRAIN & EVAL
# ======================
def train_and_evaluate(json_file_path, seq_len_override=None, seed=42):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    current_seq_len = seq_len_override if seq_len_override is not None else SEQ_LEN
    print(f"\n--- Training/Eval for SEQ_LEN = {current_seq_len} (Seed = {seed}) ---")

    # 1) Load and normalize
    ep_texts, ep_raw, mean, std = load_and_calibrate(json_file_path)

    # 2) Split episodes into train_eps and test_eps
    eps = list(ep_texts.keys())
    random.shuffle(eps)
    split = int(len(eps) * TRAIN_SPLIT)
    train_eps, test_eps = eps[:split], eps[split:]

    # 3) Build sliding-window sequences
    X_train_full, y_train_full_z, _            = build_sequences(
        ep_texts, ep_raw, mean, std, current_seq_len, train_eps
    )
    X_test, y_test_z, y_test_raw_targets = build_sequences(
        ep_texts, ep_raw, mean, std, current_seq_len, test_eps
    )

    # 4) Train/val split on training windows
    train_ds_full = ProsodyDataset(X_train_full, y_train_full_z)
    val_size      = int(len(train_ds_full) * VAL_SPLIT)
    train_size    = len(train_ds_full) - val_size
    train_ds, val_ds = random_split(train_ds_full, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(
        ProsodyDataset(X_test, y_test_z),
        batch_size=BATCH_SIZE
    )

    # 5) Model, loss, optimizer
    model     = BiLSTMProsody().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss     = float('inf')
    best_model_state  = None

    # 6) Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

        train_mse = total_train_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                total_val_loss += criterion(preds, yb).item() * xb.size(0)

        val_mse = total_val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch:02d} — Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")

        # If this is the best validation so far, store state dict in memory
        if val_mse < best_val_loss:
            best_val_loss    = val_mse
            best_model_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

    # 7) Load best model from memory (no file I/O)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state saved; using last epoch's weights.")

    # 8) Final evaluation on test set
    model.eval()
    preds_z, trues_z = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()
            preds_z.append(out)
            trues_z.append(yb.numpy())

    preds_z = np.vstack(preds_z)
    trues_z = np.vstack(trues_z)

    # Z‐score MSE/MAE
    mse_attrs_z = ((preds_z - trues_z) ** 2).mean(axis=0)
    mae_attrs_z = np.mean(np.abs(preds_z - trues_z), axis=0)

    # Convert back to raw %
    preds_raw = preds_z * std + mean
    mse_attrs_raw = ((preds_raw - y_test_raw_targets) ** 2).mean(axis=0)
    mae_attrs_raw = np.mean(np.abs(preds_raw - y_test_raw_targets), axis=0)

    print("\n--- Z‐score Metrics on Test Set ---")
    print(f"  Pitch MSE (z):  {mse_attrs_z[0]:.4f}")
    print(f"  Volume MSE (z): {mse_attrs_z[1]:.4f}")
    print(f"  Rate MSE (z):   {mse_attrs_z[2]:.4f}")
    print(f"  Avg   MSE (z):  {mse_attrs_z.mean():.4f}")

    print("\n--- Raw % Metrics on Test Set ---")
    print(f"  Pitch MAE (%):  {mae_attrs_raw[0]:.4f}")
    print(f"  Volume MAE (%): {mae_attrs_raw[1]:.4f}")
    print(f"  Rate MAE (%):   {mae_attrs_raw[2]:.4f}")
    print(f"  Avg   MAE (%):  {mae_attrs_raw.mean():.4f}")

    print(f"\n  Pitch MSE (%):  {mse_attrs_raw[0]:.4f}")
    print(f"  Volume MSE (%): {mse_attrs_raw[1]:.4f}")
    print(f"  Rate MSE (%):   {mse_attrs_raw[2]:.4f}")
    print(f"  Avg   MSE (%):  {mse_attrs_raw.mean():.4f}")


if __name__ == '__main__':

    json_path = "Code/ssml_models/jonah/bdd.json"

    sequence_lengths = [1, 2, 3, 4]
    master_seed = 123

    for s_len in sequence_lengths:
        train_and_evaluate(json_path, seq_len_override=s_len, seed=master_seed)

