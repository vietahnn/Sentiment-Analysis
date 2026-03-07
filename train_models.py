"""
Core model components for Vietnamese fintech sentiment experiments.
Provides data loading, datasets, model definitions, and train/eval utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from transformers import AutoModel


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_sentiment_label(value):
    """Map raw sentiment text into class ids: neg=0, neu=1, pos=2."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if text in {"negative", "neg", "0", "-1"}:
        return 0
    if text in {"neutral", "neu", "1"}:
        return 1
    if text in {"positive", "pos", "2"}:
        return 2

    return None


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset and create fields used by training scripts."""
    df = pd.read_excel(file_path)

    if "content" not in df.columns:
        raise ValueError("Dataset must contain 'content' column")

    if "sentiment_label" not in df.columns:
        if "sentiment" not in df.columns:
            raise ValueError("Dataset must contain either 'sentiment_label' or 'sentiment'")
        df["sentiment_label"] = df["sentiment"].apply(_normalize_sentiment_label)

    # Drop rows that cannot be mapped to one of 3 sentiment labels.
    df = df[df["sentiment_label"].isin([0, 1, 2])].copy()
    df["sentiment_label"] = df["sentiment_label"].astype(int)

    # Optional columns used by other scripts.
    if "sentiment" not in df.columns:
        inv_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        df["sentiment"] = df["sentiment_label"].map(inv_map)

    if "score" in df.columns and "rating_label" not in df.columns:
        df["rating_label"] = pd.to_numeric(df["score"], errors="coerce").fillna(3).astype(int) - 1
        df["rating_label"] = df["rating_label"].clip(0, 4)

    if "aspect_categories" not in df.columns:
        df["aspect_categories"] = "General"

    aspect_categories = [
        "Transaction",
        "UI/UX",
        "Security",
        "Customer Support",
        "Fee",
        "Promotion",
        "Savings",
        "General",
    ]

    def encode_aspects(aspect_text):
        if pd.isna(aspect_text):
            return [0] * len(aspect_categories)

        raw_parts = [part.strip() for part in str(aspect_text).split("|")]
        encoded = [1 if category in raw_parts else 0 for category in aspect_categories]
        return encoded

    df["aspect_encoded"] = df["aspect_categories"].apply(encode_aspects)

    df = df.reset_index(drop=True)
    return df


def preprocess_text(text) -> str:
    """Simple normalization suitable for app-review text."""
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    teen_code_map = {
        " ko ": " khong ",
        " k ": " khong ",
        " dc ": " duoc ",
        " dk ": " duoc ",
        " vs ": " voi ",
        " cx ": " cung ",
        " nt ": " nhan tin ",
        " ib ": " nhan tin ",
    }

    text = f" {text} "
    for old, new in teen_code_map.items():
        text = text.replace(old, new)

    return " ".join(text.split())


def get_class_weights(labels):
    """Compute balanced class weights for CrossEntropyLoss."""
    labels = np.asarray(labels)
    classes = np.array([0, 1, 2])

    class_counts = np.array([(labels == class_id).sum() for class_id in classes], dtype=np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)

    weights = class_counts.sum() / (len(classes) * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class SentimentDataset(Dataset):
    """Dataset for token-index sequence models (LSTM/BiLSTM/GRU)."""

    def __init__(self, texts, labels, vocab=None, max_len=128):
        self.texts = [preprocess_text(value) for value in texts]
        self.labels = np.asarray(labels).astype(int)
        self.max_len = max_len
        self.vocab = vocab if vocab is not None else self._build_vocab()

    def _build_vocab(self):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        next_index = 2

        for text in self.texts:
            for token in text.split():
                if token not in vocab:
                    vocab[token] = next_index
                    next_index += 1

        return vocab

    def _to_sequence(self, text):
        tokens = text.split()[: self.max_len]
        ids = [self.vocab.get(token, 1) for token in tokens]

        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))

        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self._to_sequence(self.texts[idx]), dtype=torch.long)
        label = torch.tensor([int(self.labels[idx])], dtype=torch.long)
        return {"input_ids": input_ids, "labels": label}


class TransformerDataset(Dataset):
    """Dataset for transformer-based classifiers."""

    def __init__(self, texts, labels, tokenizer, max_len=128, ratings=None, aspects=None, multi_task=False):
        self.texts = [preprocess_text(value) for value in texts]
        self.labels = np.asarray(labels).astype(int)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.multi_task = multi_task
        self.ratings = ratings
        self.aspects = aspects

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor([int(self.labels[idx])], dtype=torch.long),
        }

        if self.multi_task:
            rating_value = 2 if self.ratings is None else int(self.ratings[idx])
            aspect_value = [0.0] * 8 if self.aspects is None else self.aspects[idx]
            item["ratings"] = torch.tensor([rating_value], dtype=torch.long)
            item["aspects"] = torch.tensor(aspect_value, dtype=torch.float)

        return item


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(self.dropout(hidden[-1]))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.bilstm(embedded)
        hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden_state))


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return self.fc(self.dropout(hidden[-1]))


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(self.dropout(context))


class PhoBERTClassifier(nn.Module):
    def __init__(self, model_name="vinai/phobert-base", num_classes=3, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


class XLMRClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", num_classes=3, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


def train_epoch(model, dataloader, optimizer, criterion, target_device, model_type="rnn"):
    """One training epoch for either RNN or transformer model."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad()

        labels = batch["labels"].squeeze(1).to(target_device)

        if model_type == "rnn":
            logits = model(batch["input_ids"].to(target_device))
        else:
            logits = model(
                batch["input_ids"].to(target_device),
                batch["attention_mask"].to(target_device),
            )

        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def evaluate(model, dataloader, criterion, target_device, model_type="rnn"):
    """Evaluate model and return summary metrics used by report tables."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].squeeze(1).to(target_device)

            if model_type == "rnn":
                logits = model(batch["input_ids"].to(target_device))
            else:
                logits = model(
                    batch["input_ids"].to(target_device),
                    batch["attention_mask"].to(target_device),
                )

            loss = criterion(logits, labels)
            total_loss += float(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)

    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1_scores.tolist(),
        },
        "predictions": all_preds,
        "true_labels": all_labels,
    }
