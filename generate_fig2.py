import re
import json
import os
import matplotlib.pyplot as plt

LOG_FILE = "result.txt"
HISTORY_FILE = "training_history.json"
OUT_FILE = "training_loss.png"
OUT_FILE_PDF = "training_loss.pdf"

# Parse blocks like: "... Training LSTM with none strategy..."
# Some logs include time prefixes (e.g., "2867.4s 398 ..."), so use search instead of start anchor.
start_pat = re.compile(r"Training\s+(.+?)\s+with\s+(\w+)\s+strategy\.\.\.")
epoch_pat = re.compile(
    r"Epoch\s+(\d+)/(\d+):\s+Train\s+Loss=([0-9.]+),\s+Val\s+Loss=([0-9.]+)"
)

tracks = {}


def parse_from_history(path):
    parsed = {}
    strategies = ["class_weights", "hybrid", "smote", "none"]
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    for key, payload in history.items():
        model_name = None
        strategy = None
        for s in strategies:
            suffix = f"_{s}"
            if key.endswith(suffix):
                model_name = key[: -len(suffix)]
                strategy = s
                break
        if model_name is None:
            continue

        train_losses = payload.get("train_losses", [])
        val_losses = payload.get("val_losses", [])
        if train_losses and val_losses:
            parsed[(model_name, strategy)] = {
                "train": train_losses,
                "val": val_losses,
            }
    return parsed


def parse_from_log(path):
    parsed = {}
    current_key = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m_start = start_pat.search(line)
            if m_start:
                model_name = m_start.group(1)
                strategy = m_start.group(2)
                current_key = (model_name, strategy)
                parsed[current_key] = {"train": [], "val": []}
                continue

            m_epoch = epoch_pat.search(line)
            if m_epoch and current_key is not None:
                train_loss = float(m_epoch.group(3))
                val_loss = float(m_epoch.group(4))
                parsed[current_key]["train"].append(train_loss)
                parsed[current_key]["val"].append(val_loss)
    return parsed


if os.path.exists(HISTORY_FILE):
    tracks = parse_from_history(HISTORY_FILE)
    print(f"Loaded curves from {HISTORY_FILE}")
else:
    tracks = parse_from_log(LOG_FILE)
    print(f"Loaded curves from {LOG_FILE}")

# Choose curves for Figure 2 across 4 main models.
# Prefer "none" for consistency, then fallback to another available strategy.
selected_models = ["LSTM", "BiLSTM", "PhoBERT", "XLM-RoBERTa"]
strategy_preference = ["none", "hybrid", "smote", "class_weights"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes = axes.flatten()

for ax, model_name in zip(axes, selected_models):
    chosen_key = None
    for strategy in strategy_preference:
        key = (model_name, strategy)
        if key in tracks and tracks[key]["train"]:
            chosen_key = key
            break

    if chosen_key is None:
        print(f"Parsed 0 epochs for {model_name} (all strategies missing)")
        ax.set_title(f"{model_name} (missing)")
        ax.axis("off")
        continue

    print(
        f"Parsed {len(tracks[chosen_key]['train'])} epochs for "
        f"{chosen_key[0]} ({chosen_key[1]})"
    )

    train_losses = tracks[chosen_key]["train"]
    val_losses = tracks[chosen_key]["val"]
    epochs = list(range(1, len(train_losses) + 1))

    ax.plot(epochs, train_losses, label="Train loss", linewidth=1.8)
    ax.plot(epochs, val_losses, label="Val loss", linewidth=1.8)
    ax.set_title(f"{chosen_key[0]} ({chosen_key[1]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("Training and validation loss curves", fontsize=13)
plt.savefig(OUT_FILE, dpi=300)
plt.savefig(OUT_FILE_PDF)
print(f"Saved {OUT_FILE}")
print(f"Saved {OUT_FILE_PDF}")
