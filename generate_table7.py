import os
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel

from train_models import (
    load_data,
    SentimentDataset,
    TransformerDataset,
    LSTMClassifier,
    device,
)
from run_experiments import resample_text_data, split_train_val_chronological, CONFIG


CHECKPOINT_DIR = "models"
DATA_PATH = "data.xlsx"
OUTPUT_PATH = "results_table7_error_examples.csv"

EXAMPLES = [
    ("nuot tien khach hang", "Neg"),
    ("Good app nay, vay go VayTotNhat nhan 5tr", "Pos"),
    ("SV phi linh tinh tum lum", "Neg"),
    ("ok", "Neu"),
    ("Khong muon cho nguoi dung voucher...", "Neg"),
]

LABEL_MAP = {0: "Neg", 1: "Neu", 2: "Pos"}


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


def build_hybrid_vocab(df):
    train_size = int(0.8 * len(df))
    trainval_df = df.iloc[:train_size].reset_index(drop=True)
    model_train_df, _ = split_train_val_chronological(trainval_df, CONFIG["val_ratio_within_train"])

    x_train = model_train_df["content"].values
    y_train = model_train_df["sentiment_label"].values
    x_res, y_res = resample_text_data(x_train, y_train, "hybrid")

    dataset = SentimentDataset(x_res, y_res, max_len=CONFIG["max_len"])
    return dataset.vocab


def predict_lstm(texts, vocab):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "LSTM_hybrid_best.pt")
    model = LSTMClassifier(len(vocab), CONFIG["embedding_dim"], CONFIG["hidden_dim"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = SentimentDataset(texts, [0] * len(texts), vocab=vocab, max_len=CONFIG["max_len"])
    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            logits = model(item["input_ids"].unsqueeze(0).to(device))
            pred = int(torch.argmax(logits, dim=1).item())
            preds.append(LABEL_MAP[pred])
    return preds


def predict_transformer(texts, model_name):
    if model_name == "PhoBERT":
        ckpt = os.path.join(CHECKPOINT_DIR, "PhoBERT_hybrid_best.pt")
        backbone_name = "vinai/phobert-base"
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        model = LocalPhoBERTClassifier(backbone_name).to(device)
    else:
        ckpt = os.path.join(CHECKPOINT_DIR, "XLM-RoBERTa_hybrid_best.pt")
        backbone_name = "xlm-roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        model = LocalXLMRClassifier(backbone_name).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    dataset = TransformerDataset(texts, [0] * len(texts), tokenizer, max_len=CONFIG["max_len"])
    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            logits = model(
                item["input_ids"].unsqueeze(0).to(device),
                item["attention_mask"].unsqueeze(0).to(device),
            )
            pred = int(torch.argmax(logits, dim=1).item())
            preds.append(LABEL_MAP[pred])
    return preds


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    for name in ["LSTM_hybrid_best.pt", "PhoBERT_hybrid_best.pt", "XLM-RoBERTa_hybrid_best.pt"]:
        p = os.path.join(CHECKPOINT_DIR, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint: {p}")

    df = load_data(DATA_PATH)
    vocab = build_hybrid_vocab(df)

    texts = [text for text, _ in EXAMPLES]
    true_labels = [label for _, label in EXAMPLES]

    lstm_preds = predict_lstm(texts, vocab)
    phobert_preds = predict_transformer(texts, "PhoBERT")
    xlmr_preds = predict_transformer(texts, "XLM-RoBERTa")

    out_df = pd.DataFrame(
        {
            "Review Text": texts,
            "True": true_labels,
            "LSTM": lstm_preds,
            "PhoBERT": phobert_preds,
            "XLM-R": xlmr_preds,
        }
    )
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(out_df)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
