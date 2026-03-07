"""
Training Pipeline for Vietnamese Fintech Sentiment Analysis
Trains LSTM, BiLSTM, GRU, LSTM+Attention, PhoBERT, XLM-RoBERTa
with various imbalance handling strategies
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, balanced_accuracy_score, mean_absolute_error
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ====================
# Data Preprocessing
# ====================

def load_data(file_path):
    """Load and preprocess data from Excel"""
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} reviews")
    
    # Map sentiment to numeric labels
    sentiment_map = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    
    # Map aspects to multi-hot encoding (8 aspect categories)
    aspect_categories = ['Transaction', 'UI/UX', 'Security', 'Customer Support', 
                        'Fee', 'Promotion', 'Savings', 'General']
    
    def encode_aspects(aspect_str):
        if pd.isna(aspect_str):
            return [0] * 8
        aspects = str(aspect_str).split('|')
        encoded = [0] * 8
        for i, cat in enumerate(aspect_categories):
            if cat in aspects:
                encoded[i] = 1
        return encoded
    
    df['aspect_encoded'] = df['aspect_categories'].apply(encode_aspects)
    
    # Rating as 0-4 (1-5 stars mapped to 0-4)
    df['rating_label'] = df['score'] - 1
    
    return df


def preprocess_text(text):
    """Basic Vietnamese text preprocessing"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Teen code normalization
    teen_code_map = {
        ' k ': ' không ',
        ' ko ': ' không ',
        ' đc ': ' được ',
        ' dc ': ' được ',
        ' cx ': ' cũng ',
        ' j ': ' gì ',
        ' tl ': ' trả lời ',
        ' sv ': ' sử dụng ',
        ' vs ': ' với ',
        ' mk ': ' mình ',
        ' t ': ' tôi ',
        ' m ': ' mày ',
        ' đ ': ' đ ',
    }
    
    for old, new in teen_code_map.items():
        text = text.replace(old, new)
    
    return text.strip()


def get_class_weights(labels):
    """Calculate class weights for imbalanced data"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return torch.FloatTensor(weights).to(device)


# ====================
# Dataset Classes
# ====================

class SentimentDataset(Dataset):
    """Dataset for RNN models (LSTM, BiLSTM, GRU)"""
    
    def __init__(self, texts, labels, vocab=None, max_len=128):
        self.texts = [preprocess_text(t) for t in texts]
        self.labels = labels
        self.max_len = max_len
        
        # Build vocabulary
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
    
    def build_vocab(self):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for text in self.texts:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab
    
    def text_to_sequence(self, text):
        tokens = text.split()[:self.max_len]
        seq = [self.vocab.get(token, 1) for token in tokens]
        # Pad sequence
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        return seq
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        seq = self.text_to_sequence(self.texts[idx])
        return {
            'input_ids': torch.LongTensor(seq),
            'labels': torch.LongTensor([self.labels[idx]])
        }


class TransformerDataset(Dataset):
    """Dataset for Transformer models (PhoBERT, XLM-R)"""
    
    def __init__(self, texts, labels, tokenizer, max_len=128, 
                 ratings=None, aspects=None, multi_task=False):
        self.texts = [preprocess_text(t) for t in texts]
        self.labels = labels
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
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.LongTensor([self.labels[idx]])
        }
        
        if self.multi_task:
            item['ratings'] = torch.LongTensor([self.ratings[idx]])
            item['aspects'] = torch.FloatTensor(self.aspects[idx])
        
        return item


# ====================
# Model Architectures
# ====================

class LSTMClassifier(nn.Module):
    """LSTM model for sentiment classification"""
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state
        output = self.dropout(hidden[-1])
        return self.fc(output)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM model"""
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                             bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.bilstm(embedded)
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(hidden)
        return self.fc(output)


class GRUClassifier(nn.Module):
    """GRU model for sentiment classification"""
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_classes=3, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        output = self.dropout(hidden[-1])
        return self.fc(output)


class LSTMAttentionClassifier(nn.Module):
    """BiLSTM with self-attention mechanism"""
    
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, 
                 num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                             bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.bilstm(embedded)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.dropout(context)
        return self.fc(output)


class PhoBERTClassifier(nn.Module):
    """PhoBERT for sentiment classification"""
    
    def __init__(self, model_name='vinai/phobert-base', num_classes=3, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class XLMRClassifier(nn.Module):
    """XLM-RoBERTa for sentiment classification"""
    
    def __init__(self, model_name='xlm-roberta-base', num_classes=3, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class MultiTaskTransformer(nn.Module):
    """Multi-task learning: Sentiment + Rating + Aspect"""
    
    def __init__(self, model_name='vinai/phobert-base', dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific heads
        self.sentiment_head = nn.Linear(768, 3)  # 3 sentiment classes
        self.rating_head = nn.Linear(768, 5)     # 5 rating classes (1-5 stars)
        self.aspect_head = nn.Linear(768, 8)     # 8 aspect categories
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        
        sentiment_logits = self.sentiment_head(pooled)
        rating_logits = self.rating_head(pooled)
        aspect_logits = self.aspect_head(pooled)
        
        return sentiment_logits, rating_logits, aspect_logits


# ====================
# Training Functions
# ====================

def train_epoch(model, dataloader, optimizer, criterion, device, model_type='rnn'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions, true_labels = [], []
    
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        
        if model_type == 'rnn':
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].squeeze(1).to(device)  # squeeze only dim 1
            outputs = model(inputs)
        else:  # transformer
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].squeeze(1).to(device)  # squeeze only dim 1
            outputs = model(inputs, attention_mask)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return avg_loss, f1


def evaluate(model, dataloader, criterion, device, model_type='rnn'):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            if model_type == 'rnn':
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].squeeze(1).to(device)  # squeeze only dim 1
                outputs = model(inputs)
            else:  # transformer
                inputs = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].squeeze(1).to(device)  # squeeze only dim 1
                outputs = model(inputs, attention_mask)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    metrics = {
        'loss': avg_loss,
        'macro_f1': f1_score(true_labels, predictions, average='macro'),
        'weighted_f1': f1_score(true_labels, predictions, average='weighted'),
        'accuracy': accuracy_score(true_labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(true_labels, predictions)
    }
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=[0, 1, 2]
    )
    
    metrics['per_class'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist()
    }
    
    metrics['predictions'] = predictions
    metrics['true_labels'] = true_labels
    
    return metrics


def get_imbalanced_data(X, y, strategy='none'):
    """Apply imbalance handling strategy"""
    if strategy == 'none':
        return X, y
    
    elif strategy == 'smote':
        smote = SMOTE(random_state=SEED, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X.reshape(-1, 1), y)
        return X_res.flatten(), y_res
    
    elif strategy == 'undersample':
        rus = RandomUnderSampler(random_state=SEED, sampling_strategy={0: 2000})
        X_res, y_res = rus.fit_resample(X.reshape(-1, 1), y)
        return X_res.flatten(), y_res
    
    elif strategy == 'hybrid':
        # First undersample majority, then SMOTE minority
        rus = RandomUnderSampler(random_state=SEED, sampling_strategy={0: 2500})
        X_res, y_res = rus.fit_resample(X.reshape(-1, 1), y)
        smote = SMOTE(random_state=SEED, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_res, y_res)
        return X_res.flatten(), y_res
    
    return X, y


# ====================
# Main Training Script
# ====================

def main():
    # Load data
    print("Loading data...")
    df = load_data('data.xlsx')
    
    # Split data chronologically (80/20)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Save results
    results = {}
    
    # Imbalance strategies
    strategies = ['none', 'class_weights', 'smote', 'hybrid']
    
    # Train models
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Strategy: {strategy}")
        print(f"{'='*50}")
        
        # TODO: Implement training for each model type
        # This is a template - you'll need to implement full training loops
        
        pass
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete! Results saved to results.json")


if __name__ == '__main__':
    main()
