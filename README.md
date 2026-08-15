# Injury Detection from Running and Walking Data

Machine-learning experiments for injury classification from motion data. The repository includes classical ML and sequence-model approaches:

- Random Forest
- XGBoost
- SVM
- LSTM
- Temporal CNN (TCN)
- Transformer Encoder
- CNN-LSTM
- GRU
- TimesNet
- PatchTST
- Informer

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run commands from the repository root.

## Dataset

Running dataset:

```bash
pip install gdown
gdown 163iij4KxowSRwIFtdFZ-Uc2KkxoqRcv0
unzip running.zip
```

Walking dataset:

```bash
gdown 1CDlCb95Xuy5A3ZWUkuBM2cjf1o4F99zY
unzip walking.zip
```

## Train

### Classical models

Use `--loader_workers` to parallelize feature loading when appropriate.

```bash
# 1) Random Forest
python src/train.py --model rf --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking

# 2) SVM
python src/train.py --model svm --csv run_data_meta_upd.csv --data_dir walking --motion_key walking --loader_workers 8

# 3) XGBoost
python src/train.py --model xgb --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking
```

### Deep models

GPU is recommended. For a lighter CPU run, start with `--max_len 1500 --batch_size 16 --epochs 10`.

```bash
# 4) LSTM
python src/train.py --model lstm --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10

# 5) TCN
python src/train.py --model tcn --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10

# 6) GRU
python src/train.py --model gru --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10

# 7) CNN-LSTM
python src/train.py --model cnn_lstm --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10

# 8) Transformer Encoder
python src/train.py --model transformer --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10

# 9) TimesNet
python src/train.py --model timesnet --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10

# 10) PatchTST
python src/train.py --model patchtst --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10

# 11) Informer
python src/train.py --model informer --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
```

## Predict

Using a trained model:

```bash
python src/predict.py --model_dir outputs/rf --motion_key walking --input_dir walk_my_test --out_csv preds_rf.csv
```

Using a ready-trained model:

```bash
python src/predict.py --model_dir models/classic_walk/rf --motion_key walking --input_dir walk_my_test --out_csv preds_rf.csv
```

## Repository layout

```text
.
├── src/                 # training, inference, analysis and preprocessing scripts
├── models/              # saved trained models
├── outputs_run/         # running experiment outputs
├── outputs_walk/        # walking experiment outputs
├── 10npy_run/           # prepared NPY running samples
├── ce/                  # extracted feature artifacts
├── multi/               # multiclass experiment assets
├── docs/                # generated/reference documentation
├── legacy/              # retained non-runtime legacy files
├── requirements.txt
└── README.md
```

## Google Colab

[gpu-trained GRU on the running data](https://colab.research.google.com/drive/1FMXT6evpgevoWK_hIyTztyvKdsg9AznN?usp=sharing)
