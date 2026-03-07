"""
Generate figures for paper from training results
- Figure 1: Training/validation loss curves
- Figure 2: Confusion matrices for best models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_training_curves():
    """Generate Figure 1: Training and validation loss curves"""
    
    print("Generating Figure 1: Training loss curves...")
    
    # Load training history
    with open('training_history.json', 'r') as f:
        history = json.load(f)
    
    # Select models to plot (best performing ones with hybrid strategy)
    models_to_plot = [
        'LSTM_hybrid',
        'BiLSTM_hybrid',
        'PhoBERT_hybrid',
        'XLM-RoBERTa_hybrid'
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    for model_key in models_to_plot:
        if model_key in history:
            model_name = model_key.split('_')[0]
            train_losses = history[model_key]['train_losses']
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, marker='o', markersize=3, 
                    linewidth=2, label=model_name, alpha=0.7)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    for model_key in models_to_plot:
        if model_key in history:
            model_name = model_key.split('_')[0]
            val_losses = history[model_key]['val_losses']
            epochs = range(1, len(val_losses) + 1)
            ax2.plot(epochs, val_losses, marker='s', markersize=3, 
                    linewidth=2, label=model_name, alpha=0.7)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss Convergence', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/training_loss.png', dpi=300, bbox_inches='tight')
    print("✓ Saved to figures/training_loss.png")
    plt.close()


def plot_confusion_matrices():
    """Generate Figure 2: Confusion matrices for best models"""
    
    print("\nGenerating Figure 2: Confusion matrices...")
    
    # This requires predictions from best models
    # For demonstration, we'll create placeholder matrices
    # In actual implementation, load saved predictions
    
    # Load test results (you need to save these during training)
    try:
        # Example: Load predictions for best models
        models = ['LSTM', 'BiLSTM', 'PhoBERT', 'XLM-RoBERTa']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        class_names = ['Negative', 'Neutral', 'Positive']
        
        for idx, model_name in enumerate(models):
            # Load model predictions (placeholder - implement actual loading)
            # For now, create example confusion matrix
            
            # Example confusion matrix (replace with actual predictions)
            cm = np.array([
                [750, 50, 30],   # Negative
                [80, 40, 20],    # Neutral  
                [30, 10, 90]     # Positive
            ])
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx], cbar=True, square=True)
            
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved to figures/confusion_matrix.png")
        plt.close()
        
    except Exception as e:
        print(f"Error generating confusion matrices: {e}")
        print("Note: Run training first to generate predictions")


def plot_aspect_performance():
    """Generate aspect-level performance comparison"""
    
    print("\nGenerating aspect performance chart...")
    
    # Example data (replace with actual results from Table 4)
    aspects = ['Transaction', 'UI/UX', 'Security', 'Support', 'Fee', 'Promotion', 'General']
    
    # Placeholder data - replace with actual
    data = {
        'LSTM': [0.65, 0.70, 0.60, 0.62, 0.68, 0.72, 0.75],
        'BiLSTM': [0.68, 0.72, 0.63, 0.65, 0.70, 0.74, 0.77],
        'PhoBERT': [0.78, 0.82, 0.75, 0.77, 0.80, 0.83, 0.85],
        'XLM-R': [0.76, 0.80, 0.73, 0.75, 0.78, 0.81, 0.83]
    }
    
    x = np.arange(len(aspects))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (model, scores) in enumerate(data.items()):
        ax.bar(x + i*width, scores, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Aspect Category', fontsize=12)
    ax.set_ylabel('Macro-F1 Score', fontsize=12)
    ax.set_title('Performance by Aspect Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 0.9])
    
    plt.tight_layout()
    plt.savefig('figures/aspect_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved to figures/aspect_performance.png")
    plt.close()


def plot_imbalance_strategy_comparison():
    """Compare imbalance handling strategies"""
    
    print("\nGenerating imbalance strategy comparison...")
    
    # Load results
    df = pd.read_csv('results_table2_overall.csv')
    
    # Filter for one model (e.g., PhoBERT) to show strategy comparison
    model_df = df[df['Model'] == 'PhoBERT']
    
    strategies = model_df['Strategy'].values
    macro_f1 = model_df['Macro-F1'].values
    balanced_acc = model_df['Balanced Acc'].values
    
    x = np.arange(len(strategies))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, macro_f1, width, label='Macro-F1', alpha=0.8)
    bars2 = ax.bar(x + width/2, balanced_acc, width, label='Balanced Accuracy', alpha=0.8)
    
    ax.set_xlabel('Imbalance Handling Strategy', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('PhoBERT: Impact of Imbalance Handling Strategies', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved to figures/strategy_comparison.png")
    plt.close()


def generate_all_figures():
    """Generate all figures for paper"""
    
    print("="*60)
    print("Generating All Figures for Paper")
    print("="*60)
    
    import os
    os.makedirs('figures', exist_ok=True)
    
    # Generate main figures
    plot_training_curves()
    plot_confusion_matrices()
    
    # Additional analysis figures
    try:
        plot_aspect_performance()
        plot_imbalance_strategy_comparison()
    except Exception as e:
        print(f"Could not generate additional figures: {e}")
    
    print("\n" + "="*60)
    print("Figure generation complete!")
    print("="*60)
    print("\nGenerated figures:")
    print("  1. figures/training_loss.png (for paper)")
    print("  2. figures/confusion_matrix.png (for paper)")
    print("  3. figures/aspect_performance.png (supplementary)")
    print("  4. figures/strategy_comparison.png (supplementary)")


if __name__ == '__main__':
    generate_all_figures()
