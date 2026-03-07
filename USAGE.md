# 🚀 USAGE GUIDE - Command Line Arguments

All scripts now support flexible path configuration via command-line arguments.

## 📌 Quick Examples

### On Kaggle
```bash
# Training with Kaggle paths
python run_experiments.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --output-dir /kaggle/working/results

# Quick test on Kaggle
python demo_quick.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --samples 500

# Test models on Kaggle  
python test_model.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --model PhoBERT \
  --strategy hybrid

# Generate figures on Kaggle
python generate_figures.py \
  --input-dir /kaggle/working/results \
  --output-dir /kaggle/working/figures
```

### On Local Machine
```bash
# Training locally (default paths)
python run_experiments.py

# Or with custom paths
python run_experiments.py \
  --data-path C:/Data/momo_data.xlsx \
  --models-dir ./checkpoints \
  --output-dir ./outputs
```

---

## 📖 Detailed Options

### 1️⃣ run_experiments.py (Main Training)

```bash
python run_experiments.py --help
```

**Arguments:**
- `--data-path` - Path to dataset file (default: `data.xlsx`)
- `--models-dir` - Directory to save checkpoints (default: `models`)
- `--output-dir` - Directory to save results (default: `.`)
- `--epochs` - Number of training epochs (default: 50)
- `--patience` - Early stopping patience (default: 5)
- `--batch-size-rnn` - Batch size for RNN models (default: 16)
- `--batch-size-transformer` - Batch size for Transformers (default: 8)

**Examples:**
```bash
# Reduced epochs for quick test
python run_experiments.py --epochs 10 --patience 3

# Smaller batch size for limited GPU memory
python run_experiments.py --batch-size-transformer 4

# Complete Kaggle example
python run_experiments.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --output-dir /kaggle/working \
  --epochs 30 \
  --batch-size-transformer 4
```

---

### 2️⃣ demo_quick.py (Quick Test)

```bash
python demo_quick.py --help
```

**Arguments:**
- `--data-path` - Path to dataset file (default: `data.xlsx`)
- `--samples` - Number of samples to test (default: 200)

**Examples:**
```bash
# Default 200 samples
python demo_quick.py

# Test with 1000 samples
python demo_quick.py --samples 1000

# Kaggle path
python demo_quick.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --samples 500
```

---

### 3️⃣ test_model.py (Test Checkpoints)

```bash
python test_model.py --help
```

**Arguments:**
- `--data-path` - Path to dataset file (default: `data.xlsx`)
- `--models-dir` - Directory containing checkpoints (default: `models`)
- `--model` - Test specific model (e.g., `LSTM`, `PhoBERT`) (default: test all)
- `--strategy` - Test specific strategy (default: `hybrid`)

**Examples:**
```bash
# Test all available models
python test_model.py

# Test specific model
python test_model.py --model PhoBERT --strategy hybrid

# Test on Kaggle
python test_model.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --model LSTM \
  --strategy none
```

---

### 4️⃣ generate_figures.py (Create Figures)

```bash
python generate_figures.py --help
```

**Arguments:**
- `--input-dir` - Directory containing results files (default: `.`)
- `--output-dir` - Directory to save figures (default: `figures`)

**Examples:**
```bash
# Default paths
python generate_figures.py

# Custom paths
python generate_figures.py \
  --input-dir ./results \
  --output-dir ./paper_figures

# Kaggle paths
python generate_figures.py \
  --input-dir /kaggle/working \
  --output-dir /kaggle/working/figures
```

---

## 🎯 Complete Kaggle Workflow

```bash
# 1. Quick test first
python demo_quick.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --samples 200

# 2. Run full training (reduced epochs for Kaggle's 9h limit)
python run_experiments.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --output-dir /kaggle/working \
  --epochs 30 \
  --patience 5 \
  --batch-size-transformer 4

# 3. Test best model
python test_model.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --model PhoBERT \
  --strategy hybrid

# 4. Generate figures
python generate_figures.py \
  --input-dir /kaggle/working \
  --output-dir /kaggle/working/figures

# 5. Download results
# - /kaggle/working/results_table2_overall.csv
# - /kaggle/working/results_table3_perclass.csv
# - /kaggle/working/models/PhoBERT_hybrid_best.pt
# - /kaggle/working/figures/*.png
```

---

## 💡 Tips

### Reduce Training Time on Kaggle
```bash
python run_experiments.py \
  --epochs 20 \              # Reduce from 50 to 20
  --patience 3 \             # Reduce from 5 to 3
  --batch-size-transformer 4  # Reduce from 8 to 4 for GPU memory
```

### Train Only One Model (modify script)
Edit `run_experiments.py` manually to comment out unwanted models:
```python
rnn_models = ['LSTM']  # Instead of ['LSTM', 'BiLSTM', 'GRU']
strategies = ['hybrid']  # Instead of ['none', 'class_weights', 'smote', 'hybrid']
```

### Check Model Availability
```bash
ls /kaggle/working/models/
# or
python -c "import os; print(os.listdir('/kaggle/working/models'))"
```

---

## 🐛 Troubleshooting

### FileNotFoundError: data.xlsx
```bash
# Check dataset path on Kaggle
ls /kaggle/input/

# Find data.xlsx
find /kaggle/input/ -name "data.xlsx"

# Use the correct path
python run_experiments.py --data-path /kaggle/input/YOUR-DATASET/data.xlsx
```

### Missing results files for generate_figures.py
```bash
# Check if training_history.json exists
ls /kaggle/working/training_history.json

# If not, run experiments first
python run_experiments.py --output-dir /kaggle/working

# Then generate figures
python generate_figures.py --input-dir /kaggle/working
```

---

## 📝 Default Values Summary

| Script | Argument | Default Value |
|--------|----------|---------------|
| All | `--data-path` | `data.xlsx` |
| run_experiments.py | `--models-dir` | `models` |
| run_experiments.py | `--output-dir` | `.` (current directory) |
| run_experiments.py | `--epochs` | 50 |
| run_experiments.py | `--patience` | 5 |
| demo_quick.py | `--samples` | 200 |
| test_model.py | `--models-dir` | `models` |
| test_model.py | `--strategy` | `hybrid` |
| generate_figures.py | `--input-dir` | `.` |
| generate_figures.py | `--output-dir` | `figures` |
