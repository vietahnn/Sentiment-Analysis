# Vietnamese Fintech Sentiment Analysis - Kaggle Training

This repository contains training code for Vietnamese sentiment analysis on Momo fintech app reviews.

## 📋 Files Included

### Core Training Files
- `train_models.py` - Model architectures and training utilities
- `run_experiments.py` - Main training script (20 experiments)
- `generate_figures.py` - Generate paper figures
- `export_latex_tables.py` - Export results to LaTeX format
- `requirements.txt` - Python dependencies
- `test_model.py` - Test trained models (optional)
- `demo_quick.py` - Quick pipeline test (optional)

### What's NOT Included (upload separately)
- `data.xlsx` - Dataset (4,325 Momo reviews) - **Upload to Kaggle Dataset**
- `models/` - Will be created during training
- `figures/` - Will be created when generating figures
- `*.csv`, `*.json` - Results will be generated

## 🚀 Setup on Kaggle

### 1. Create New Notebook
- Go to Kaggle → New Notebook
- Settings → Enable GPU/TPU (recommended)
- Add Dataset: Upload `data.xlsx` as separate dataset

### 2. Install Dependencies
```python
!pip install -r requirements.txt
```

### 3. Upload Dataset
Link your Kaggle dataset or upload `data.xlsx` to input folder:
```python
import shutil
# Copy from Kaggle input to working directory
shutil.copy('/kaggle/input/momo-reviews/data.xlsx', 'data.xlsx')
```

### 4. Run Training
```python
# Quick test (2-3 minutes, 200 samples)
!python demo_quick.py

# Full training (3-5 hours with GPU)
!python run_experiments.py
```

### 5. Download Results
After training completes, download:
- `models/*.pt` - Trained checkpoints
- `results_table2_overall.csv` - Overall performance
- `results_table3_perclass.csv` - Per-class metrics
- `training_history.json` - Training curves

### 6. Generate Figures
```python
!python generate_figures.py
# Downloads: training_loss.png, confusion_matrix.png
```

### 7. Export LaTeX Tables
```python
!python export_latex_tables.py
# Downloads: latex_tables.txt
```

## 📊 Training Details

- **Models:** LSTM, BiLSTM, GRU, PhoBERT, XLM-RoBERTa
- **Strategies:** None, Class Weights, SMOTE, Hybrid
- **Total Experiments:** 20 (5 models × 4 strategies)
- **Dataset Split:** 80% train (3,460) / 20% test (865)
- **Expected Time:** 3-5 hours (GPU) / 10-15 hours (CPU)

## 📝 Configuration

Edit `run_experiments.py` to modify:
```python
CONFIG = {
    'batch_size_rnn': 16,
    'batch_size_transformer': 8,
    'num_epochs': 50,
    'patience': 5,
    'learning_rate_rnn': 0.001,
    'learning_rate_transformer': 2e-5,
}
```

## 🔍 Monitor Progress

```python
# Check if training is running
import os
print("Checkpoints:", len([f for f in os.listdir('models') if f.endswith('.pt')]))

# View results
import pandas as pd
df = pd.read_csv('results_table2_overall.csv')
print(df.sort_values('Macro-F1', ascending=False))
```

## 💾 Save Results

```python
from IPython.display import FileLink, FileLinks

# Download all results
FileLinks('.', recursive=True)

# Download specific files
FileLink('results_table2_overall.csv')
FileLink('models/PhoBERT_hybrid_best.pt')
```

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch size in `run_experiments.py`:
```python
'batch_size_transformer': 4,  # Reduce from 8 to 4
```

### Missing Dependencies
```python
!pip install imbalanced-learn==0.11.0
!pip install transformers==4.30.2
```

### Dataset Not Found
Ensure `data.xlsx` is in working directory:
```python
import os
print(os.path.exists('data.xlsx'))
```

## 📈 Expected Results

Best models typically achieve:
- **Macro-F1:** 0.65-0.75
- **Weighted-F1:** 0.75-0.85
- **Balanced Accuracy:** 0.70-0.80

## 📧 Contact

For issues or questions about the training code, please open an issue on GitHub.
