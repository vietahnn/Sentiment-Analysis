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
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from train_models import *
import json
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--data-path', type=str, default='/kaggle/input/datasets/nguyenthanhvung/sentiment-analysis/data.xlsx',
                       help='Path to dataset file on Kaggle input storage')
    parser.add_argument('--models-dir', type=str, default='/kaggle/working/Sentiment-Analysis/models',
                       help='Directory containing model checkpoints on Kaggle working storage')
    parser.add_argument('--model', type=str, default=None,
                       help='Test specific model (e.g., LSTM, PhoBERT)')
    parser.add_argument('--strategy', type=str, default='hybrid',
                       help='Test specific strategy (default: hybrid)')
    args = parser.parse_args()

print("="*60)
print("MODEL TESTING SCRIPT")
print("="*60)
print(f"Data path: {args.data_path}")
print(f"Models dir: {args.models_dir}")

# Load data
print("\nLoading data...")
df = load_data(args.data_path)

# Split (same as training)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size].reset_index(drop=True)
test_df = df.iloc[train_size:].reset_index(drop=True)

X_train_text = train_df['content'].values
y_train = train_df['sentiment_label'].values
X_test_text = test_df['content'].values
y_test = test_df['sentiment_label'].values


def split_train_val_chronological(train_slice, val_ratio=0.1):
    """Match the run_experiments split before building RNN vocab."""
    val_size = max(1, int(len(train_slice) * val_ratio))
    if val_size >= len(train_slice):
        val_size = max(1, len(train_slice) - 1)
    return train_slice[:-val_size], train_slice[-val_size:]


def resample_text_data(texts, labels, strategy='none'):
    """Same text-safe resampling logic used in run_experiments.py."""
    X = np.array(texts, dtype=object)
    y = np.array(labels)

    if strategy in ['none', 'class_weights']:
        return X, y

    if strategy == 'smote':
        ros = RandomOverSampler(random_state=SEED)
        X_res, y_res = ros.fit_resample(X.reshape(-1, 1), y)
        return X_res.flatten(), y_res

    if strategy == 'hybrid':
        class_counts = pd.Series(y).value_counts().to_dict()
        majority_class = max(class_counts, key=class_counts.get)
        minority_counts = [count for cls, count in class_counts.items() if cls != majority_class]

        if minority_counts:
            target_majority = max(minority_counts) * 2
            target_majority = min(target_majority, class_counts[majority_class])
            rus = RandomUnderSampler(
                random_state=SEED,
                sampling_strategy={majority_class: target_majority}
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

print(f"Test set size: {len(test_df)} samples")
print(f"Sentiment distribution:")
print(test_df['sentiment'].value_counts())


def test_rnn_model(model_name, strategy='hybrid'):
    """Test a trained RNN model"""
    
    print(f"\n{'='*50}")
    print(f"Testing {model_name} ({strategy} strategy)")
    print(f"{'='*50}")
    
    # Check if model checkpoint exists
    checkpoint_path = os.path.join(args.models_dir, f'{model_name}_{strategy}_best.pt')
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found:{checkpoint_path}")
        return None
    
    # Recreate the exact train vocabulary path used in training.
    model_train_texts, _ = split_train_val_chronological(X_train_text, val_ratio=0.1)
    model_train_labels, _ = split_train_val_chronological(y_train, val_ratio=0.1)
    X_train_resampled, y_train_resampled = resample_text_data(model_train_texts, model_train_labels, strategy)

    # Create datasets
    train_dataset = SentimentDataset(X_train_resampled, y_train_resampled, max_len=128)
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
    checkpoint_path = os.path.join(args.models_dir, f'{model_name}_{strategy}_best.pt')
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
    if not os.path.exists(args.models_dir):
        print(f"❌ No models directory found: {args.models_dir}")
        print("   Train models first with: python run_experiments.py")
        return
    
    checkpoints = [f for f in os.listdir(args.models_dir) if f.endswith('.pt')]
    
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


# Test based on arguments
if args.model:
    print(f"\nTesting specific model: {args.model} ({args.strategy})")
    if args.model in ['LSTM', 'BiLSTM', 'GRU', 'LSTM+Attention']:
        test_rnn_model(args.model, args.strategy)
    elif args.model in ['PhoBERT', 'XLM-RoBERTa']:
        test_transformer_model(args.model, args.strategy)
    else:
        print(f"Unknown model: {args.model}")
else:
    # Test all available models
    test_all_models()
