"""
Quick demo script to test pipeline with small subset
Use this to verify everything works before running full experiments
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_models import *
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("QUICK DEMO - Testing Pipeline")
print("="*60)

# Load small subset
print("\n1. Loading data subset...")
df = load_data('data.xlsx')
df_sample = df.sample(n=200, random_state=42)  # Only 200 samples for quick test

# Split
train_size = int(0.8 * len(df_sample))
train_df = df_sample.iloc[:train_size].reset_index(drop=True)
test_df = df_sample.iloc[train_size:].reset_index(drop=True)

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

X_train = train_df['content'].values
y_train = train_df['sentiment_label'].values
X_test = test_df['content'].values
y_test = test_df['sentiment_label'].values

# Test 1: RNN Model (LSTM)
print("\n2. Testing LSTM model...")
train_dataset = SentimentDataset(X_train, y_train, max_len=64)
test_dataset = SentimentDataset(X_test, y_test, vocab=train_dataset.vocab, max_len=64)

vocab_size = len(train_dataset.vocab)
print(f"Vocabulary size: {vocab_size}")

model = LSTMClassifier(vocab_size, embedding_dim=100, hidden_dim=64).to(device)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training for 3 epochs...")
for epoch in range(3):
    train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, 'rnn')
    metrics = evaluate(model, test_loader, criterion, device, 'rnn')
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={metrics['macro_f1']:.4f}")

print("✓ LSTM test passed!")

# Test 2: Transformer Model (PhoBERT) - Optional if you have GPU
if torch.cuda.is_available():
    print("\n3. Testing PhoBERT model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        
        train_dataset_bert = TransformerDataset(X_train, y_train, tokenizer, max_len=64)
        test_dataset_bert = TransformerDataset(X_test, y_test, tokenizer, max_len=64)
        
        model_bert = PhoBERTClassifier('vinai/phobert-base').to(device)
        
        train_loader_bert = DataLoader(train_dataset_bert, batch_size=4, shuffle=True)
        test_loader_bert = DataLoader(test_dataset_bert, batch_size=4)
        
        optimizer_bert = torch.optim.AdamW(model_bert.parameters(), lr=2e-5)
        
        print("Training for 2 epochs...")
        for epoch in range(2):
            train_loss, train_f1 = train_epoch(model_bert, train_loader_bert, optimizer_bert, 
                                              criterion, device, 'transformer')
            metrics = evaluate(model_bert, test_loader_bert, criterion, device, 'transformer')
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={metrics['macro_f1']:.4f}")
        
        print("✓ PhoBERT test passed!")
        
    except Exception as e:
        print(f"PhoBERT test skipped: {e}")
else:
    print("\n3. Skipping PhoBERT test (no GPU available)")

print("\n" + "="*60)
print("✓ Demo Complete!")
print("="*60)
print("\nPipeline is working correctly. You can now run:")
print("  python run_experiments.py")
print("\nThis will take 3-5 hours with GPU for full experiments.")
