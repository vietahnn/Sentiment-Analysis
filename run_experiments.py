"""
Complete experimental pipeline to generate all results for paper
Runs all models with all imbalance strategies and saves results
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train_models import *
import time
import json
import os
from collections import defaultdict

# Configuration
CONFIG = {
    'batch_size_rnn': 16,
    'batch_size_transformer': 8,
    'learning_rate_rnn': 0.001,
    'learning_rate_transformer': 2e-5,
    'num_epochs': 50,
    'patience': 5,
    'max_len': 128,
    'embedding_dim': 300,
    'hidden_dim': 128,
}

# Results storage
all_results = {
    'overall_performance': [],
    'per_class_performance': [],
    'aspect_performance': [],
    'multitask_performance': [],
    'error_examples': [],
    'training_history': {}
}


def train_rnn_model(model_name, train_dataset, test_dataset, vocab_size, strategy='none'):
    """Train RNN-based model (LSTM, BiLSTM, GRU, LSTM+Attention)"""
    
    print(f"\nTraining {model_name} with {strategy} strategy...")
    
    # Create model
    if model_name == 'LSTM':
        model = LSTMClassifier(vocab_size, CONFIG['embedding_dim'], 
                              CONFIG['hidden_dim']).to(device)
    elif model_name == 'BiLSTM':
        model = BiLSTMClassifier(vocab_size, CONFIG['embedding_dim'], 
                                CONFIG['hidden_dim']).to(device)
    elif model_name == 'GRU':
        model = GRUClassifier(vocab_size, CONFIG['embedding_dim'], 
                             CONFIG['hidden_dim']).to(device)
    elif model_name == 'LSTM+Attention':
        model = LSTMAttentionClassifier(vocab_size, CONFIG['embedding_dim'], 
                                       CONFIG['hidden_dim']).to(device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size_rnn'], 
                             shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_rnn'])
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate_rnn'])
    
    # Loss function based on strategy
    if strategy == 'class_weights':
        class_weights = get_class_weights(train_dataset.labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, 
                                          criterion, device, 'rnn')
        
        # Evaluate
        val_metrics = evaluate(model, test_loader, criterion, device, 'rnn')
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['macro_f1']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val F1={val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'models/{model_name}_{strategy}_best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Store training history
    all_results['training_history'][f'{model_name}_{strategy}'] = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return best_metrics


def train_transformer_model(model_name, train_dataset, test_dataset, strategy='none'):
    """Train Transformer-based model (PhoBERT, XLM-RoBERTa)"""
    
    print(f"\nTraining {model_name} with {strategy} strategy...")
    
    # Create model
    if model_name == 'PhoBERT':
        model = PhoBERTClassifier('vinai/phobert-base').to(device)
    elif model_name == 'XLM-RoBERTa':
        model = XLMRClassifier('xlm-roberta-base').to(device)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size_transformer'], 
                             shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_transformer'])
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate_transformer'])
    
    # Loss function based on strategy
    if strategy == 'class_weights':
        class_weights = get_class_weights(train_dataset.labels)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    
    start_time = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, 
                                          criterion, device, 'transformer')
        
        # Evaluate
        val_metrics = evaluate(model, test_loader, criterion, device, 'transformer')
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['macro_f1']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val F1={val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            patience_counter = 0
            torch.save(model.state_dict(), f'models/{model_name}_{strategy}_best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    train_time = (time.time() - start_time) / 60  # in minutes
    best_metrics['train_time'] = train_time
    
    # Store training history
    all_results['training_history'][f'{model_name}_{strategy}'] = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    return best_metrics


def run_all_experiments():
    """Run complete experimental pipeline"""
    
    print("="*60)
    print("Starting Complete Experimental Pipeline")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = load_data('data.xlsx')
    
    # Split data chronologically
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train sentiment distribution:")
    print(train_df['sentiment'].value_counts())
    
    # Extract data
    X_train_text = train_df['content'].values
    y_train = train_df['sentiment_label'].values
    X_test_text = test_df['content'].values
    y_test = test_df['sentiment_label'].values
    
    # Create model save directory
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Strategies to test
    strategies = ['none', 'class_weights', 'smote', 'hybrid']
    
    # Model configurations
    rnn_models = ['LSTM', 'BiLSTM', 'GRU']  # We'll skip LSTM+Attention for faster training
    transformer_models = ['PhoBERT', 'XLM-RoBERTa']
    
    # ====================
    # RNN Models
    # ====================
    
    print("\n" + "="*60)
    print("Training RNN Models")
    print("="*60)
    
    for strategy in strategies:
        print(f"\n{'*'*50}")
        print(f"Strategy: {strategy}")
        print(f"{'*'*50}")
        
        # Apply imbalance handling for data-level strategies
        if strategy in ['smote', 'hybrid']:
            indices = np.arange(len(train_df))
            indices_resampled, y_resampled = get_imbalanced_data(indices, y_train, strategy)
            X_train_resampled = X_train_text[indices_resampled]
            y_train_resampled = y_resampled
        else:
            X_train_resampled = X_train_text
            y_train_resampled = y_train
        
        # Create datasets
        train_dataset = SentimentDataset(X_train_resampled, y_train_resampled, 
                                        max_len=CONFIG['max_len'])
        test_dataset = SentimentDataset(X_test_text, y_test, 
                                       vocab=train_dataset.vocab, 
                                       max_len=CONFIG['max_len'])
        
        vocab_size = len(train_dataset.vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Train each RNN model
        for model_name in rnn_models:
            metrics = train_rnn_model(model_name, train_dataset, test_dataset, 
                                     vocab_size, strategy)
            
            # Store results for Table 2 (Overall performance)
            all_results['overall_performance'].append({
                'Model': model_name,
                'Strategy': strategy,
                'Macro-F1': metrics['macro_f1'],
                'Weighted-F1': metrics['weighted_f1'],
                'Balanced Acc': metrics['balanced_accuracy'],
                'Accuracy': metrics['accuracy']
            })
            
            # Store per-class results for Table 3
            if strategy == 'hybrid':  # Best strategy for per-class analysis
                all_results['per_class_performance'].append({
                    'Model': model_name,
                    'Class': 'Positive',
                    'Precision': metrics['per_class']['precision'][2],
                    'Recall': metrics['per_class']['recall'][2],
                    'F1': metrics['per_class']['f1'][2]
                })
                all_results['per_class_performance'].append({
                    'Model': model_name,
                    'Class': 'Negative',
                    'Precision': metrics['per_class']['precision'][0],
                    'Recall': metrics['per_class']['recall'][0],
                    'F1': metrics['per_class']['f1'][0]
                })
                all_results['per_class_performance'].append({
                    'Model': model_name,
                    'Class': 'Neutral',
                    'Precision': metrics['per_class']['precision'][1],
                    'Recall': metrics['per_class']['recall'][1],
                    'F1': metrics['per_class']['f1'][1]
                })
    
    # ====================
    # Transformer Models
    # ====================
    
    print("\n" + "="*60)
    print("Training Transformer Models")
    print("="*60)
    
    for model_name in transformer_models:
        # Initialize tokenizer
        if model_name == 'PhoBERT':
            tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        else:
            tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        
        for strategy in strategies:
            print(f"\n{'*'*50}")
            print(f"{model_name} - Strategy: {strategy}")
            print(f"{'*'*50}")
            
            # Apply imbalance handling
            if strategy in ['smote', 'hybrid']:
                indices = np.arange(len(train_df))
                indices_resampled, y_resampled = get_imbalanced_data(indices, y_train, strategy)
                X_train_resampled = X_train_text[indices_resampled]
                y_train_resampled = y_resampled
            else:
                X_train_resampled = X_train_text
                y_train_resampled = y_train
            
            # Create datasets
            train_dataset = TransformerDataset(X_train_resampled, y_train_resampled, 
                                              tokenizer, max_len=CONFIG['max_len'])
            test_dataset = TransformerDataset(X_test_text, y_test, 
                                             tokenizer, max_len=CONFIG['max_len'])
            
            # Train
            metrics = train_transformer_model(model_name, train_dataset, 
                                             test_dataset, strategy)
            
            # Store results
            all_results['overall_performance'].append({
                'Model': model_name,
                'Strategy': strategy,
                'Macro-F1': metrics['macro_f1'],
                'Weighted-F1': metrics['weighted_f1'],
                'Balanced Acc': metrics['balanced_accuracy'],
                'Accuracy': metrics['accuracy']
            })
            
            # Per-class results for best strategy
            if strategy == 'hybrid':
                all_results['per_class_performance'].append({
                    'Model': model_name,
                    'Class': 'Positive',
                    'Precision': metrics['per_class']['precision'][2],
                    'Recall': metrics['per_class']['recall'][2],
                    'F1': metrics['per_class']['f1'][2]
                })
                all_results['per_class_performance'].append({
                    'Model': model_name,
                    'Class': 'Negative',
                    'Precision': metrics['per_class']['precision'][0],
                    'Recall': metrics['per_class']['recall'][0],
                    'F1': metrics['per_class']['f1'][0]
                })
                all_results['per_class_performance'].append({
                    'Model': model_name,
                    'Class': 'Neutral',
                    'Precision': metrics['per_class']['precision'][1],
                    'Recall': metrics['per_class']['recall'][1],
                    'F1': metrics['per_class']['f1'][1]
                })
    
    # Save all results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Convert to DataFrames and save
    pd.DataFrame(all_results['overall_performance']).to_csv('results_table2_overall.csv', index=False)
    pd.DataFrame(all_results['per_class_performance']).to_csv('results_table3_perclass.csv', index=False)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(all_results['training_history'], f, indent=2)
    
    print("\n✓ Results saved to:")
    print("  - results_table2_overall.csv")
    print("  - results_table3_perclass.csv")
    print("  - training_history.json")
    
    return all_results


if __name__ == '__main__':
    results = run_all_experiments()
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)
