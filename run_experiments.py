"""
Complete experimental pipeline to generate all results for paper
Runs all models with all imbalance strategies and saves results
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from train_models import *
import time
import json
import os
import gc
import argparse
from collections import defaultdict

# Configuration
CONFIG = {
    'batch_size_rnn': 16,
    'batch_size_transformer': 4,
    'learning_rate_rnn': 5e-4,
    'learning_rate_transformer': 2e-5,
    'weight_decay_rnn': 1e-5,
    'weight_decay_transformer': 1e-2,
    'label_smoothing': 0.05,
    'num_epochs': 50,
    'patience': 5,
    'max_len': 128,
    'embedding_dim': 300,
    'hidden_dim': 128,
    'val_ratio_within_train': 0.1,
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


def resample_text_data(texts, labels, strategy='none'):
    """Text-safe imbalance handling for sequence models."""
    X = np.array(texts, dtype=object)
    y = np.array(labels)

    if strategy in ['none', 'class_weights']:
        return X, y

    # For text data, use random oversampling instead of synthetic interpolation.
    if strategy == 'smote':
        ros = RandomOverSampler(random_state=SEED)
        X_res, y_res = ros.fit_resample(X.reshape(-1, 1), y)
        return X_res.flatten(), y_res

    if strategy == 'hybrid':
        class_counts = pd.Series(y).value_counts().to_dict()
        majority_class = max(class_counts, key=class_counts.get)
        minority_counts = [count for cls, count in class_counts.items() if cls != majority_class]

        if minority_counts:
            # First reduce majority pressure, then rebalance with oversampling.
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


def split_train_val_chronological(train_df, val_ratio):
    """Split training slice into train/validation while keeping temporal order."""
    val_size = max(1, int(len(train_df) * val_ratio))
    if val_size >= len(train_df):
        val_size = max(1, len(train_df) - 1)

    model_train_df = train_df.iloc[:-val_size].reset_index(drop=True)
    val_df = train_df.iloc[-val_size:].reset_index(drop=True)
    return model_train_df, val_df


def create_train_loader(dataset, batch_size, strategy):
    """Create training loader and optionally rebalance with weighted sampling."""
    labels = np.asarray(dataset.labels)
    class_counts = np.bincount(labels, minlength=3).astype(np.float32)
    class_counts[class_counts == 0] = 1.0

    if strategy == 'class_weights':
        sample_weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def save_intermediate_results(output_dir):
    """Persist partial results after each model to avoid losing long runs."""
    table2_path = os.path.join(output_dir, 'results_table2_overall.csv')
    table3_path = os.path.join(output_dir, 'results_table3_perclass.csv')
    history_path = os.path.join(output_dir, 'training_history.json')

    pd.DataFrame(all_results['overall_performance']).to_csv(table2_path, index=False)
    pd.DataFrame(all_results['per_class_performance']).to_csv(table3_path, index=False)
    with open(history_path, 'w') as f:
        json.dump(all_results['training_history'], f, indent=2)


def cleanup_memory():
    """Release Python and CUDA cached memory between long training runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_rnn_model(model_name, train_dataset, val_dataset, test_dataset, vocab_size, strategy='none'):
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
    train_loader = create_train_loader(train_dataset, CONFIG['batch_size_rnn'], strategy)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size_rnn'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_rnn'])
    
    # Setup training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate_rnn'],
        weight_decay=CONFIG['weight_decay_rnn'],
    )
    
    # Loss function based on strategy
    if strategy == 'class_weights':
        class_weights = get_class_weights(train_dataset.labels)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=CONFIG['label_smoothing'],
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    
    # Training loop
    best_f1 = -1
    best_metrics = None
    patience_counter = 0
    train_losses, val_losses = [], []
    
    model_path = os.path.join(args.models_dir, f'{model_name}_{strategy}_best.pt')

    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, 
                                          criterion, device, 'rnn')
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, 'rnn')
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['macro_f1']
        val_weighted_f1 = val_metrics['weighted_f1']
        val_balanced_acc = val_metrics['balanced_accuracy']
        val_acc = val_metrics['accuracy']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(
            f"Epoch {epoch+1}/{CONFIG['num_epochs']}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Macro-F1={val_f1:.4f}, Weighted-F1={val_weighted_f1:.4f}, "
            f"Bal-Acc={val_balanced_acc:.4f}, Acc={val_acc:.4f}"
        )
        
        # Early stopping
        if best_metrics is None or val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            patience_counter = 0
            # Save best model by validation Macro-F1
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluate best checkpoint on held-out test set
    model.load_state_dict(torch.load(model_path, map_location=device))
    best_metrics = evaluate(model, test_loader, criterion, device, 'rnn')
    best_metrics['best_val_macro_f1'] = best_f1

    # Store training history
    all_results['training_history'][f'{model_name}_{strategy}'] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_macro_f1': best_f1,
    }
    
    final_metrics = dict(best_metrics)

    # Explicit cleanup helps avoid RAM/VRAM accumulation in long experiment loops.
    del model, optimizer, criterion, train_loader, val_loader, test_loader
    cleanup_memory()

    return final_metrics


def train_transformer_model(model_name, train_dataset, val_dataset, test_dataset, strategy='none'):
    """Train Transformer-based model (PhoBERT, XLM-RoBERTa)"""
    
    print(f"\nTraining {model_name} with {strategy} strategy...")
    
    # Create model
    if model_name == 'PhoBERT':
        model = PhoBERTClassifier('vinai/phobert-base').to(device)
    elif model_name == 'XLM-RoBERTa':
        model = XLMRClassifier('xlm-roberta-base').to(device)
    
    # Create dataloaders
    train_loader = create_train_loader(train_dataset, CONFIG['batch_size_transformer'], strategy)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size_transformer'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_transformer'])
    
    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate_transformer'],
        weight_decay=CONFIG['weight_decay_transformer'],
    )
    
    # Loss function based on strategy
    if strategy == 'class_weights':
        class_weights = get_class_weights(train_dataset.labels)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=CONFIG['label_smoothing'],
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    
    # Training loop
    best_f1 = -1
    best_metrics = None
    patience_counter = 0
    train_losses, val_losses = [], []
    
    start_time = time.time()
    
    model_path = os.path.join(args.models_dir, f'{model_name}_{strategy}_best.pt')

    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, 
                                          criterion, device, 'transformer')
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device, 'transformer')
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['macro_f1']
        val_weighted_f1 = val_metrics['weighted_f1']
        val_balanced_acc = val_metrics['balanced_accuracy']
        val_acc = val_metrics['accuracy']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(
            f"Epoch {epoch+1}/{CONFIG['num_epochs']}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
            f"Macro-F1={val_f1:.4f}, Weighted-F1={val_weighted_f1:.4f}, "
            f"Bal-Acc={val_balanced_acc:.4f}, Acc={val_acc:.4f}"
        )
        
        # Early stopping
        if best_metrics is None or val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = val_metrics
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    train_time = (time.time() - start_time) / 60  # in minutes

    # Evaluate best checkpoint on held-out test set
    model.load_state_dict(torch.load(model_path, map_location=device))
    best_metrics = evaluate(model, test_loader, criterion, device, 'transformer')
    best_metrics['train_time'] = train_time
    best_metrics['best_val_macro_f1'] = best_f1
    
    # Store training history
    all_results['training_history'][f'{model_name}_{strategy}'] = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_macro_f1': best_f1,
    }
    
    final_metrics = dict(best_metrics)

    # Explicit cleanup helps avoid RAM/VRAM accumulation in long experiment loops.
    del model, optimizer, criterion, train_loader, val_loader, test_loader
    cleanup_memory()

    return final_metrics


def run_all_experiments(args):
    """Run complete experimental pipeline"""

    global all_results
    all_results = {
        'overall_performance': [],
        'per_class_performance': [],
        'aspect_performance': [],
        'multitask_performance': [],
        'error_examples': [],
        'training_history': {}
    }
    
    print("="*60)
    print("Starting Complete Experimental Pipeline")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    print(f"Data path: {args.data_path}")
    df = load_data(args.data_path)
    
    # Split data chronologically: 80% train+val, 20% held-out test.
    train_size = int(0.8 * len(df))
    trainval_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    model_train_df, val_df = split_train_val_chronological(
        trainval_df,
        CONFIG['val_ratio_within_train'],
    )

    print(f"Train: {len(model_train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("Train sentiment distribution:")
    print(model_train_df['sentiment'].value_counts())
    
    # Extract data
    X_train_text = model_train_df['content'].values
    y_train = model_train_df['sentiment_label'].values
    X_val_text = val_df['content'].values
    y_val = val_df['sentiment_label'].values
    X_test_text = test_df['content'].values
    y_test = test_df['sentiment_label'].values
    
    # Create model save directory
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Strategies to test
    strategies = ['none', 'class_weights', 'smote', 'hybrid']
    
    # Model configurations
    rnn_models = ['LSTM', 'BiLSTM', 'GRU', 'LSTM+Attention']
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
        X_train_resampled, y_train_resampled = resample_text_data(
            X_train_text, y_train, strategy
        )
        
        # Create datasets
        train_dataset = SentimentDataset(X_train_resampled, y_train_resampled, 
                                        max_len=CONFIG['max_len'])
        val_dataset = SentimentDataset(X_val_text, y_val,
                          vocab=train_dataset.vocab,
                          max_len=CONFIG['max_len'])
        test_dataset = SentimentDataset(X_test_text, y_test, 
                                       vocab=train_dataset.vocab, 
                                       max_len=CONFIG['max_len'])
        
        vocab_size = len(train_dataset.vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Train each RNN model
        for model_name in rnn_models:
            metrics = train_rnn_model(model_name, train_dataset, val_dataset, test_dataset,
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

            save_intermediate_results(args.output_dir)
            cleanup_memory()
    
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
            X_train_resampled, y_train_resampled = resample_text_data(
                X_train_text, y_train, strategy
            )
            
            # Create datasets
            train_dataset = TransformerDataset(X_train_resampled, y_train_resampled, 
                                              tokenizer, max_len=CONFIG['max_len'])
            val_dataset = TransformerDataset(X_val_text, y_val,
                                            tokenizer, max_len=CONFIG['max_len'])
            test_dataset = TransformerDataset(X_test_text, y_test, 
                                             tokenizer, max_len=CONFIG['max_len'])
            
            # Train
            metrics = train_transformer_model(model_name, train_dataset, val_dataset,
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

            save_intermediate_results(args.output_dir)
            cleanup_memory()
    
    # Save all results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Convert to DataFrames and save
    table2_path = os.path.join(args.output_dir, 'results_table2_overall.csv')
    table3_path = os.path.join(args.output_dir, 'results_table3_perclass.csv')
    history_path = os.path.join(args.output_dir, 'training_history.json')
    
    pd.DataFrame(all_results['overall_performance']).to_csv(table2_path, index=False)
    pd.DataFrame(all_results['per_class_performance']).to_csv(table3_path, index=False)
    
    # Save training history
    with open(history_path, 'w') as f:
        json.dump(all_results['training_history'], f, indent=2)
    
    print("\n✓ Results saved to:")
    print(f"  - {table2_path}")
    print(f"  - {table3_path}")
    print(f"  - {history_path}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sentiment analysis experiments')
    parser.add_argument('--data-path', type=str, default='/kaggle/input/datasets/nguyenthanhvung/sentiment-analysis/data.xlsx',
                       help='Path to dataset file on Kaggle input storage')
    parser.add_argument('--models-dir', type=str, default='/kaggle/working/Sentiment-Analysis/models',
                       help='Directory to save model checkpoints on Kaggle working storage')
    parser.add_argument('--output-dir', type=str, default='/kaggle/working/Sentiment-Analysis',
                       help='Directory to save results on Kaggle working storage')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (default: 5)')
    parser.add_argument('--batch-size-rnn', type=int, default=16,
                       help='Batch size for RNN models (default: 16)')
    parser.add_argument('--batch-size-transformer', type=int, default=4,
                       help='Batch size for Transformer models (default: 4)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation ratio within training split (default: 0.1)')
    
    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    CONFIG['num_epochs'] = args.epochs
    CONFIG['patience'] = args.patience
    CONFIG['batch_size_rnn'] = args.batch_size_rnn
    CONFIG['batch_size_transformer'] = args.batch_size_transformer
    CONFIG['val_ratio_within_train'] = args.val_ratio
    
    print("\nConfiguration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Models dir: {args.models_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Patience: {args.patience}")
    print(f"  Batch size (RNN): {args.batch_size_rnn}")
    print(f"  Batch size (Transformer): {args.batch_size_transformer}")
    print(f"  Validation ratio (within train): {args.val_ratio}")
    
    results = run_all_experiments(args)
    print("\n" + "="*60)
    print("All experiments complete!")
    print("="*60)
