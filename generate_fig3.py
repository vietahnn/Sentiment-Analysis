import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoConfig, AutoModel
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from train_models import (
    load_data,
    SentimentDataset,
    TransformerDataset,
    LSTMClassifier,
    SEED,
    device,
)
from run_experiments import CONFIG, split_train_val_chronological


DATA_PATH = "data.xlsx"
MODEL_DIR = "models"
OUT_PDF = "confusion_matrix.pdf"
OUT_PNG = "confusion_matrix.png"

CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def resample_text_data(texts, labels, strategy="none"):
    X = np.array(texts, dtype=object)
    y = np.array(labels)

    if strategy in ["none", "class_weights"]:
        return X, y

    if strategy == "smote":
        ros = RandomOverSampler(random_state=SEED)
        X_res, y_res = ros.fit_resample(X.reshape(-1, 1), y)
        return X_res.flatten(), y_res

    if strategy == "hybrid":
        class_counts = pd.Series(y).value_counts().to_dict()
        majority_class = max(class_counts, key=class_counts.get)
        minority_counts = [count for cls, count in class_counts.items() if cls != majority_class]

        if minority_counts:
            target_majority = max(minority_counts) * 2
            target_majority = min(target_majority, class_counts[majority_class])
            rus = RandomUnderSampler(
                random_state=SEED,
                sampling_strategy={majority_class: target_majority},
            )
            X_under, y_under = rus.fit_resample(X.reshape(-1, 1), y)
        else:
            X_under, y_under = X.reshape(-1, 1), y

        balanced_target = int(pd.Series(y_under).value_counts().max())
        ros_strategy = {
            cls: balanced_target
            for cls, count in pd.Series(y_under).value_counts().to_dict().items()
            if count < balanced_target
        }

        if ros_strategy:
            ros = RandomOverSampler(random_state=SEED, sampling_strategy=ros_strategy)
            X_balanced, y_balanced = ros.fit_resample(X_under, y_under)
            return X_balanced.flatten(), y_balanced

        return X_under.flatten(), y_under

    return X, y


class LocalPhoBERTClassifier(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", num_classes=3, dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_config(config)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


class LocalXLMRClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_classes=3, dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_config(config)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


def get_test_split(df):
    train_size = int(0.8 * len(df))
    trainval_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    model_train_df, _ = split_train_val_chronological(trainval_df, CONFIG["val_ratio_within_train"])
    return model_train_df, test_df


def predict_lstm_hybrid(model_train_df, test_df):
    x_train = model_train_df["content"].values
    y_train = model_train_df["sentiment_label"].values
    x_test = test_df["content"].values

    x_res, y_res = resample_text_data(x_train, y_train, strategy="hybrid")
    train_dataset = SentimentDataset(x_res, y_res, max_len=CONFIG["max_len"])
    test_dataset = SentimentDataset(x_test, [0] * len(x_test), vocab=train_dataset.vocab, max_len=CONFIG["max_len"])

    model = LSTMClassifier(len(train_dataset.vocab), CONFIG["embedding_dim"], CONFIG["hidden_dim"]).to(device)
    ckpt = os.path.join(MODEL_DIR, "LSTM_hybrid_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            logits = model(item["input_ids"].unsqueeze(0).to(device))
            preds.append(int(torch.argmax(logits, dim=1).item()))
    return np.array(preds)


def predict_transformer_hybrid(test_df, which):
    x_test = test_df["content"].values

    if which == "phobert":
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        model = LocalPhoBERTClassifier("vinai/phobert-base").to(device)
        ckpt = os.path.join(MODEL_DIR, "PhoBERT_hybrid_best.pt")
    else:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = LocalXLMRClassifier("xlm-roberta-base").to(device)
        ckpt = os.path.join(MODEL_DIR, "XLM-RoBERTa_hybrid_best.pt")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    dataset = TransformerDataset(x_test, [0] * len(x_test), tokenizer, max_len=CONFIG["max_len"])
    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            logits = model(
                item["input_ids"].unsqueeze(0).to(device),
                item["attention_mask"].unsqueeze(0).to(device),
            )
            preds.append(int(torch.argmax(logits, dim=1).item()))
    return np.array(preds)


def draw_confusion(ax, cm, title):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=9)

    return im


def main():
    df = load_data(DATA_PATH)
    model_train_df, test_df = get_test_split(df)
    y_true = test_df["sentiment_label"].values

    preds_lstm = predict_lstm_hybrid(model_train_df, test_df)
    preds_phobert = predict_transformer_hybrid(test_df, "phobert")
    preds_xlmr = predict_transformer_hybrid(test_df, "xlmr")

    cm_lstm = confusion_matrix(y_true, preds_lstm, labels=[0, 1, 2])
    cm_phobert = confusion_matrix(y_true, preds_phobert, labels=[0, 1, 2])
    cm_xlmr = confusion_matrix(y_true, preds_xlmr, labels=[0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    draw_confusion(axes[0], cm_lstm, "LSTM (hybrid)")
    draw_confusion(axes[1], cm_phobert, "PhoBERT (hybrid)")
    draw_confusion(axes[2], cm_xlmr, "XLM-RoBERTa (hybrid)")

    fig.suptitle("Confusion matrices on held-out test set", fontsize=13)
    plt.savefig(OUT_PNG, dpi=300)
    plt.savefig(OUT_PDF)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
