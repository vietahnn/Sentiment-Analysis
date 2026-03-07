# ✅ HOÀN THÀNH: Thêm Argument Parser Cho Tất Cả Scripts

## 🎯 Mục Đích
Thay vì hardcode đường dẫn `data.xlsx`, `models/`, giờ bạn có thể chỉnh đường dẫn bất kỳ qua command-line arguments.

## 📝 Files Đã Được Cập Nhật

### 1. run_experiments.py ✅
**Arguments mới:**
```bash
--data-path          # Đường dẫn dataset (default: data.xlsx)
--models-dir         # Thư mục lưu checkpoints (default: models)
--output-dir         # Thư mục lưu kết quả (default: .)
--epochs             # Số epochs (default: 50)
--patience           # Early stopping patience (default: 5)
--batch-size-rnn     # Batch size RNN (default: 16)
--batch-size-transformer # Batch size Transformer (default: 8)
```

**Cách dùng trên Kaggle:**
```bash
python run_experiments.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --output-dir /kaggle/working
```

### 2. demo_quick.py ✅
**Arguments mới:**
```bash
--data-path          # Đường dẫn dataset (default: data.xlsx)
--samples            # Số samples test (default: 200)
```

**Cách dùng:**
```bash
python demo_quick.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --samples 500
```

### 3. test_model.py ✅
**Arguments mới:**
```bash
--data-path          # Đường dẫn dataset (default: data.xlsx)
--models-dir         # Thư mục chứa checkpoints (default: models)
--model              # Test model cụ thể: LSTM, PhoBERT (default: test all)
--strategy           # Strategy: none, hybrid, etc. (default: hybrid)
```

**Cách dùng:**
```bash
# Test tất cả models
python test_model.py --data-path /kaggle/input/momo-reviews/data.xlsx

# Test model cụ thể
python test_model.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --model PhoBERT \
  --strategy hybrid
```

### 4. generate_figures.py ✅
**Arguments mới:**
```bash
--input-dir          # Thư mục chứa results files (default: .)
--output-dir         # Thư mục lưu figures (default: figures)
```

**Cách dùng:**
```bash
python generate_figures.py \
  --input-dir /kaggle/working \
  --output-dir /kaggle/working/figures
```

## 🚀 Sử Dụng Trên Kaggle

### Workflow Hoàn Chỉnh:
```bash
# 1. Quick test
python demo_quick.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --samples 200

# 2. Full training
python run_experiments.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models \
  --output-dir /kaggle/working \
  --epochs 30

# 3. Test models
python test_model.py \
  --data-path /kaggle/input/momo-reviews/data.xlsx \
  --models-dir /kaggle/working/models

# 4. Generate figures
python generate_figures.py \
  --input-dir /kaggle/working \
  --output-dir /kaggle/working/figures
```

## 📖 Xem Help
Mỗi script đều có `--help`:
```bash
python run_experiments.py --help
python demo_quick.py --help
python test_model.py --help
python generate_figures.py --help
```

## ✨ Lợi Ích

### ✅ Trước (hardcoded)
```python
# Trong code:
df = load_data('data.xlsx')  # ❌ Cố định
torch.save(model, 'models/best.pt')  # ❌ Cố định
```

**Vấn đề:** Phải sửa code mỗi khi đổi đường dẫn (local → Kaggle)

### ✅ Sau (flexible)
```bash
# Không cần sửa code, chỉ command line:
python run_experiments.py --data-path /kaggle/input/momo-reviews/data.xlsx
```

**Lợi ích:** 
- ✅ Không cần sửa code
- ✅ Dễ switch giữa local và Kaggle
- ✅ Dễ test nhiều configs khác nhau
- ✅ Professional và flexible hơn

## 📚 Tài Liệu Chi Tiết
Xem file **USAGE.md** để biết tất cả options và examples.

## 🔧 Files Đã Tạo Mới
1. **USAGE.md** - Hướng dẫn sử dụng chi tiết
2. **SUMMARY.md** - File này (tóm tắt thay đổi)

## 💾 Commit & Push
```bash
git add run_experiments.py demo_quick.py test_model.py generate_figures.py
git add USAGE.md SUMMARY.md
git commit -m "Add argparse for flexible path configuration"
git push origin main
```

## ✅ HOÀN THÀNH!
Giờ bạn có thể train trên Kaggle mà không cần sửa code, chỉ cần điều chỉnh arguments!
