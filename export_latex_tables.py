"""
Export experimental results to LaTeX table format
Ready to copy-paste into paper.tex
"""

import pandas as pd
import numpy as np


def export_table2_overall():
    """Export Table 2: Overall Performance"""
    
    print("="*60)
    print("TABLE 2: Overall Performance with Imbalance Handling")
    print("="*60)
    
    df = pd.read_csv('results_table2_overall.csv')
    
    # Group by model
    models = ['LSTM', 'BiLSTM', 'GRU', 'PhoBERT', 'XLM-RoBERTa']
    strategies = ['none', 'class_weights', 'smote', 'hybrid']
    
    latex_rows = []
    
    for model in models:
        model_df = df[df['Model'] == model]
        
        for i, strategy in enumerate(strategies):
            row_df = model_df[model_df['Strategy'] == strategy]
            
            if not row_df.empty:
                row = row_df.iloc[0]
                
                # Format strategy name
                strat_name = {
                    'none': 'No balancing',
                    'class_weights': 'Class weights',
                    'smote': 'SMOTE',
                    'hybrid': 'Hybrid'
                }[strategy]
                
                # Format row
                if i == 0:
                    latex_row = f"\\multirow{{4}}{{*}}{{{model}}} \n"
                else:
                    latex_row = ""
                
                latex_row += f"& {strat_name} & {row['Macro-F1']:.4f} & {row['Weighted-F1']:.4f} & {row['Balanced Acc']:.4f} & {row['Accuracy']:.4f} \\\\"
                
                latex_rows.append(latex_row)
        
        latex_rows.append("\\midrule")
    
    # Remove last midrule
    if latex_rows and latex_rows[-1] == "\\midrule":
        latex_rows.pop()
    
    print("\nLaTeX code for Table 2:")
    print("\n".join(latex_rows))
    
    return latex_rows


def export_table3_perclass():
    """Export Table 3: Per-Class Performance"""
    
    print("\n" + "="*60)
    print("TABLE 3: Per-Class Performance")
    print("="*60)
    
    df = pd.read_csv('results_table3_perclass.csv')
    
    models = ['LSTM', 'BiLSTM', 'PhoBERT', 'XLM-RoBERTa']
    classes = ['Positive', 'Negative', 'Neutral']
    
    latex_rows = []
    
    for model in models:
        model_df = df[df['Model'] == model]
        
        for i, cls in enumerate(classes):
            row_df = model_df[model_df['Class'] == cls]
            
            if not row_df.empty:
                row = row_df.iloc[0]
                
                if i == 0:
                    latex_row = f"\\multirow{{3}}{{*}}{{{model}}} \n"
                else:
                    latex_row = ""
                
                latex_row += f"& {cls} & {row['Precision']:.4f} & {row['Recall']:.4f} & {row['F1']:.4f} \\\\"
                
                latex_rows.append(latex_row)
        
        latex_rows.append("\\midrule")
    
    # Remove last midrule
    if latex_rows and latex_rows[-1] == "\\midrule":
        latex_rows.pop()
    
    print("\nLaTeX code for Table 3:")
    print("\n".join(latex_rows))
    
    return latex_rows


def format_best_results():
    """Find and highlight best results"""
    
    print("\n" + "="*60)
    print("BEST RESULTS SUMMARY")
    print("="*60)
    
    df = pd.read_csv('results_table2_overall.csv')
    
    # Find best overall
    best_macro = df.loc[df['Macro-F1'].idxmax()]
    best_weighted = df.loc[df['Weighted-F1'].idxmax()]
    best_balanced = df.loc[df['Balanced Acc'].idxmax()]
    
    print(f"\nBest Macro-F1: {best_macro['Model']} ({best_macro['Strategy']}) = {best_macro['Macro-F1']:.4f}")
    print(f"Best Weighted-F1: {best_weighted['Model']} ({best_weighted['Strategy']}) = {best_weighted['Weighted-F1']:.4f}")
    print(f"Best Balanced Acc: {best_balanced['Model']} ({best_balanced['Strategy']}) = {best_balanced['Balanced Acc']:.4f}")
    
    # Compare strategies
    print("\n" + "-"*60)
    print("Strategy Comparison (Average across all models):")
    print("-"*60)
    
    strategy_avg = df.groupby('Strategy')[['Macro-F1', 'Weighted-F1', 'Balanced Acc']].mean()
    print(strategy_avg.to_string())
    
    # Compare models
    print("\n" + "-"*60)
    print("Model Comparison (Best strategy for each):")
    print("-"*60)
    
    best_per_model = df.loc[df.groupby('Model')['Macro-F1'].idxmax()]
    print(best_per_model[['Model', 'Strategy', 'Macro-F1', 'Weighted-F1', 'Balanced Acc']].to_string(index=False))


def create_latex_snippet():
    """Create complete LaTeX snippet for copy-paste"""
    
    output_file = 'latex_tables.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% LaTeX Tables - Generated from Experimental Results\n")
        f.write("% Copy and paste into paper.tex\n\n")
        
        # Table 2
        f.write("% TABLE 2: Overall Performance\n")
        f.write("% Replace empty cells in paper.tex with these values:\n\n")
        table2_rows = export_table2_overall()
        f.write("\n".join(table2_rows))
        
        f.write("\n\n")
        
        # Table 3
        f.write("% TABLE 3: Per-Class Performance\n")
        f.write("% Replace empty cells in paper.tex with these values:\n\n")
        table3_rows = export_table3_perclass()
        f.write("\n".join(table3_rows))
        
        f.write("\n\n")
        
        # Analysis
        f.write("% BEST RESULTS FOR CONCLUSION SECTION:\n")
        df = pd.read_csv('results_table2_overall.csv')
        best = df.loc[df['Macro-F1'].idxmax()]
        f.write(f"% Best Model: {best['Model']} with {best['Strategy']} strategy\n")
        f.write(f"% Macro-F1: {best['Macro-F1']:.4f}\n")
        f.write(f"% Weighted-F1: {best['Weighted-F1']:.4f}\n")
        f.write(f"% Balanced Accuracy: {best['Balanced Acc']:.4f}\n")
    
    print(f"\n✓ Complete LaTeX snippets saved to: {output_file}")


def main():
    """Generate all LaTeX exports"""
    
    print("\n" + "="*70)
    print("EXPORTING RESULTS TO LATEX FORMAT")
    print("="*70 + "\n")
    
    try:
        # Export tables
        export_table2_overall()
        export_table3_perclass()
        
        # Format summary
        format_best_results()
        
        # Create file
        create_latex_snippet()
        
        print("\n" + "="*70)
        print("✓ Export Complete!")
        print("="*70)
        print("\nNext steps:")
        print("1. Open latex_tables.txt")
        print("2. Copy the LaTeX code")
        print("3. Paste into corresponding tables in paper.tex")
        print("4. Update [To be filled] placeholders in Conclusion section")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run experiments first:")
        print("  python run_experiments.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == '__main__':
    main()
