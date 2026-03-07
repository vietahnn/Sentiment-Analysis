# Vietnamese Fintech Sentiment Analysis - Experimental Pipeline

Pipeline code để train models và tạo số liệu cho paper "Comparative Analysis of Deep Learning Models for Aspect-Based Sentiment Analysis of Vietnamese Fintech Reviews"

## Files Structure

```
research2/
├── data.xlsx                      # Dataset (4,325 Momo reviews)
├── paper.tex                      # LaTeX paper file
├── requirements.txt               # Python dependencies
├── train_models.py               # Model definitions và training functions
├── run_experiments.py            # Main experimental pipeline
├── generate_figures.py           # Generate figures cho paper
├── export_latex_tables.py        # Export results to LaTeX format
├── models/                       # Saved trained models
├── figures/                      # Generated figures
└── results/                      # CSV results files
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note về VnCoreNLP**: Cần download VnCoreNLP jar file:
```bash
mkdir vncorenlp
cd vncorenlp
wget https://github.com/vncorenlp/VnCoreNLP/raw/master/VnCoreNLP-1.1.1.jar
```

### 2. Data Preparation

Đảm bảo file `data.xlsx` có các columns:
- `content`: Review text
- `score`: Rating (1-5 stars)
- `sentiment`: Sentiment label (Positive/Negative/Neutral)
- `aspect_categories`: Aspect annotations (pipe-separated)

## Running Experiments

### Step 1: Run Complete Experimental Pipeline

```bash
python run_experiments.py
```

Việc này sẽ:
- Train 3 RNN models (LSTM, BiLSTM, GRU) với 4 imbalance strategies = 12 experiments
- Train 2 Transformer models (PhoBERT, XLM-RoBERTa) với 4 strategies = 8 experiments  
- **Tổng cộng: 20 experiments**
- Output: 
  - `results_table2_overall.csv` - Overall performance metrics
  - `results_table3_perclass.csv` - Per-class metrics
  - `training_history.json` - Training/validation losses
  - Saved models in `models/` directory

**Thời gian ước tính:**
- RNN models: ~10-15 phút/model
- Transformer models: ~30-45 phút/model (có GPU)
- **Tổng: ~3-5 giờ với GPU, ~10-15 giờ với CPU**

### Step 2: Generate Figures

```bash
python generate_figures.py
```

Output:
- `figures/training_loss.png` - Figure 1 cho paper
- `figures/confusion_matrix.png` - Figure 2 cho paper
- Additional analysis figures

### Step 3: Export to LaTeX Format

```bash
python export_latex_tables.py
```

Output:
- `latex_tables.txt` - LaTeX code để copy vào paper.tex

## Filling Paper Tables

Sau khi chạy xong experiments, bạn có các files:

### Table 2: Overall Performance
```csv
Model,Strategy,Macro-F1,Weighted-F1,Balanced Acc,Accuracy
LSTM,none,0.45,0.75,0.52,0.78
LSTM,class_weights,0.58,0.79,0.65,0.80
...
```

Copy số liệu từ CSV vào Table 2 trong paper.tex

### Table 3: Per-Class Performance  
```csv
Model,Class,Precision,Recall,F1
LSTM,Positive,0.68,0.52,0.59
LSTM,Negative,0.89,0.94,0.91
...
```

Copy vào Table 3 trong paper.tex

### Figures
Copy files từ `figures/` directory:
- `training_loss.png` → paper directory
- `confusion_matrix.png` → paper directory

## Quick Start (Simplified Version)

Nếu muốn test nhanh với subset nhỏ:

```python
# Trong run_experiments.py, thay đổi CONFIG:
CONFIG = {
    'num_epochs': 10,  # Giảm từ 50 → 10
    'patience': 3,     # Giảm từ 5 → 3
}

# Chỉ train 1-2 models để test
rnn_models = ['LSTM']  # Thay vì ['LSTM', 'BiLSTM', 'GRU']
transformer_models = ['PhoBERT']  # Thay vì cả PhoBERT và XLM-R
```

Thời gian: ~30 phút

## Expected Results Format

### Table 2 Example:
| Model | Strategy | Macro-F1 | Weighted-F1 | Bal. Acc | Acc |
|-------|----------|----------|-------------|----------|-----|
| LSTM | none | 0.45 | 0.75 | 0.52 | 0.78 |
| LSTM | class_weights | 0.58 | 0.79 | 0.65 | 0.80 |
| LSTM | smote | 0.62 | 0.77 | 0.68 | 0.79 |
| LSTM | hybrid | **0.65** | **0.80** | **0.71** | **0.81** |
| PhoBERT | hybrid | **0.78** | **0.88** | **0.82** | **0.89** |

### Table 3 Example:
| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|-----|
| PhoBERT | Positive | 0.82 | 0.75 | 0.78 |
| PhoBERT | Negative | 0.91 | 0.95 | 0.93 |
| PhoBERT | Neutral | 0.65 | 0.58 | 0.61 |

## Troubleshooting

### CUDA Out of Memory
```python
# Giảm batch size trong CONFIG
CONFIG = {
    'batch_size_transformer': 4,  # Từ 8 → 4
}
```

### VnCoreNLP Error
Bỏ qua preprocessing nâng cao, chỉ dùng basic preprocessing

### Missing Dependencies
```bash
pip install --upgrade transformers torch
```

## Notes

1. **GPU Recommended**: Training Transformers rất chậm trên CPU
2. **Disk Space**: Models chiếm ~500MB mỗi checkpoint
3. **Reproducibility**: Set SEED=42 để kết quả consistent

## Contact

Nếu có lỗi, check:
1. Data format đúng chưa (data.xlsx)
2. GPU memory đủ không
3. Dependencies install đầy đủ chưa

## Citation

```bibtex
@inproceedings{your-paper-2026,
  title={Comparative Analysis of Deep Learning Models for Aspect-Based Sentiment Analysis of Vietnamese Fintech Reviews},
  author={Your Name et al.},
  year={2026}
}
```
