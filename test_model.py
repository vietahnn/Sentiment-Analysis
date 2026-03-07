"""
Test trained models on test set
Load saved checkpoints and evaluate performance
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_models import *
import json
import os

print("="*60)
print("MODEL TESTING SCRIPT")
print("="*60)

# Load data
print("\nLoading data...")
df = load_data('data.xlsx')

# Split (same as training)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].reset_index(drop=True)
test_df = df.iloc[train_size:].reset_index(drop=True)

X_train_text = train_df['content'].values
y_train = train_df['sentiment_label'].values
X_test_text = test_df['content'].values
y_test = test_df['sentiment_label'].values

print(f"Test set size: {len(test_df)} samples")
print(f"Sentiment distribution:")
print(test_df['sentiment'].value_counts())


def test_rnn_model(model_name, strategy='hybrid'):
    """Test a trained RNN model"""
    
    print(f"\n{'='*50}")
    print(f"Testing {model_name} ({strategy} strategy)")
    print(f"{'='*50}")
    
    # Check if model checkpoint exists
    checkpoint_path = f'models/{model_name}_{strategy}_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Create datasets
    train_dataset = SentimentDataset(X_train_text, y_train, max_len=128)
    test_dataset = SentimentDataset(X_test_text, y_test, 
                                   vocab=train_dataset.vocab, max_len=128)
    vocab_size = len(train_dataset.vocab)
    
    # Create model
    if model_name == 'LSTM':
        model = LSTMClassifier(vocab_size, 300, 128).to(device)
    elif model_name == 'BiLSTM':
        model = BiLSTMClassifier(vocab_size, 300, 128).to(device)
    elif model_name == 'GRU':
        model = GRUClassifier(vocab_size, 300, 128).to(device)
    elif model_name == 'LSTM+Attention':
        model = LSTMAttentionClassifier(vocab_size, 300, 128).to(device)
    else:
        print(f"Unknown model: {model_name}")
        return None
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Evaluate
    test_loader = DataLoader(test_dataset, batch_size=16)
    criterion = nn.CrossEntropyLoss()
    
    metrics = evaluate(model, test_loader, criterion, device, 'rnn')
    
    # Print results
    print(f"\n📊 Test Results:")
    print(f"   Macro-F1:       {metrics['macro_f1']:.4f}")
    print(f"   Weighted-F1:    {metrics['weighted_f1']:.4f}")
    print(f"   Balanced Acc:   {metrics['balanced_accuracy']:.4f}")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    
    print(f"\n📋 Per-class Performance:")
    classes = ['Negative', 'Neutral', 'Positive']
    for i, cls in enumerate(classes):
        print(f"   {cls:8s}: P={metrics['per_class']['precision'][i]:.4f} "
              f"R={metrics['per_class']['recall'][i]:.4f} "
              f"F1={metrics['per_class']['f1'][i]:.4f}")
    
    return metrics


def test_transformer_model(model_name, strategy='hybrid'):
    """Test a trained Transformer model"""
    
    print(f"\n{'='*50}")
    print(f"Testing {model_name} ({strategy} strategy)")
    print(f"{'='*50}")
    
    # Check if model checkpoint exists
    checkpoint_path = f'models/{model_name}_{strategy}_best.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    # Initialize tokenizer
    if model_name == 'PhoBERT':
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        model = PhoBERTClassifier('vinai/phobert-base').to(device)
    elif model_name == 'XLM-RoBERTa':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        model = XLMRClassifier('xlm-roberta-base').to(device)
    else:
        print(f"Unknown model: {model_name}")
        return None
    
    # Create dataset
    test_dataset = TransformerDataset(X_test_text, y_test, tokenizer, max_len=128)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Evaluate
    test_loader = DataLoader(test_dataset, batch_size=8)
    criterion = nn.CrossEntropyLoss()
    
    metrics = evaluate(model, test_loader, criterion, device, 'transformer')
    
    # Print results
    print(f"\n📊 Test Results:")
    print(f"   Macro-F1:       {metrics['macro_f1']:.4f}")
    print(f"   Weighted-F1:    {metrics['weighted_f1']:.4f}")
    print(f"   Balanced Acc:   {metrics['balanced_accuracy']:.4f}")
    print(f"   Accuracy:       {metrics['accuracy']:.4f}")
    
    print(f"\n📋 Per-class Performance:")
    classes = ['Negative', 'Neutral', 'Positive']
    for i, cls in enumerate(classes):
        print(f"   {cls:8s}: P={metrics['per_class']['precision'][i]:.4f} "
              f"R={metrics['per_class']['recall'][i]:.4f} "
              f"F1={metrics['per_class']['f1'][i]:.4f}")
    
    return metrics


def test_all_models():
    """Test all available trained models"""
    
    print("\n" + "="*60)
    print("TESTING ALL AVAILABLE MODELS")
    print("="*60)
    
    # Check which models are available
    if not os.path.exists('models'):
        print("❌ No models directory found. Train models first!")
        return
    
    checkpoints = [f for f in os.listdir('models') if f.endswith('.pt')]
    
    if not checkpoints:
        print("❌ No trained models found. Train models first!")
        print("   Run: python run_experiments.py")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for cp in sorted(checkpoints):
        print(f"   ✓ {cp}")
    
    # Test each model
    results = {}
    
    # RNN models
    rnn_models = ['LSTM', 'BiLSTM', 'GRU', 'LSTM+Attention']
    for model_name in rnn_models:
        for strategy in ['none', 'class_weights', 'smote', 'hybrid']:
            checkpoint = f'{model_name}_{strategy}_best.pt'
            if checkpoint in checkpoints:
                metrics = test_rnn_model(model_name, strategy)
                if metrics:
                    results[f'{model_name}_{strategy}'] = metrics
    
    # Transformer models
    transformer_models = ['PhoBERT', 'XLM-RoBERTa']
    for model_name in transformer_models:
        for strategy in ['none', 'class_weights', 'smote', 'hybrid']:
            checkpoint = f'{model_name}_{strategy}_best.pt'
            if checkpoint in checkpoints:
                metrics = test_transformer_model(model_name, strategy)
                if metrics:
                    results[f'{model_name}_{strategy}'] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - BEST MODELS BY MACRO-F1")
    print("="*60)
    
    if results:
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['macro_f1'], 
                              reverse=True)
        
        print(f"\n{'Rank':<6} {'Model':<25} {'Macro-F1':<12} {'Weighted-F1':<12} {'Bal-Acc':<12}")
        print("-" * 70)
        for rank, (model_key, metrics) in enumerate(sorted_results[:10], 1):
            print(f"{rank:<6} {model_key:<25} {metrics['macro_f1']:<12.4f} "
                  f"{metrics['weighted_f1']:<12.4f} {metrics['balanced_accuracy']:<12.4f}")
    else:
        print("No results to display")
    
    return results


if __name__ == '__main__':
    # Option 1: Test all available models
    test_all_models()
    
    # Option 2: Test specific model (uncomment to use)
    # test_rnn_model('LSTM', 'hybrid')
    # test_transformer_model('PhoBERT', 'hybrid')
