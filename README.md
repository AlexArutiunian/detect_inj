# 11 ml approaches for task to detect injury on running and walking dataset

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

## Upload dataset

```bash
pip install gdown
```

Upload [running.zip](https://drive.google.com/file/d/163iij4KxowSRwIFtdFZ-Uc2KkxoqRcv0)

```bash
gdown 163iij4KxowSRwIFtdFZ-Uc2KkxoqRcv0
unzip running.zip
```

Upload [walking.zip](https://drive.google.com/file/d/1CDlCb95Xuy5A3ZWUkuBM2cjf1o4F99zY)

```bash
gdown 1CDlCb95Xuy5A3ZWUkuBM2cjf1o4F99zY
unzip walking.zip
```

## Train

### Classic (CPU is OK)

This flag is for parallel boost by multicore (8 or 8/2=4 cores)
```bash
--loader_workers 8 
```
```bash
# 1) Random Forest
python train.py --model rf  --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking
# 2) SVM
python train.py --model svm --csv run_data_meta_upd.csv --data_dir walking --motion_key walking --loader_workers 8
# 3) XGBoost
python train.py --model xgb --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking
```

### Deep (GPU recommended; if CPU: start with `--max_len 1500 --batch_size 16 --epochs 10`):

```bash
# 4) LSTM
python train.py --model lstm --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
# 5) TCN
python train.py --model tcn  --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
# 6) GRU
python train.py --model gru  --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
# 7) CNN-LSTM
python train.py --model cnn_lstm --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
# 8) Transformer Encoder
python train.py --model transformer --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
# 9) TimesNet (lite)
python train.py --model timesnet --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
# 10) PatchTST (lite)
python train.py --model patchtst --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
```
```bash
# 10) Informer (lite)

python train.py --model informer --csv walk_data_meta_upd.csv --data_dir walking --motion_key walking --max_len 1500 --batch_size 16 --epochs 10
```

## Predict (using the trained model)

```bash
python predict.py --model_dir outputs/rf  --motion_key walking --input_dir walk_my_test --out_csv preds_rf.csv
```

Or using the ready-trained models

```bash
python predict.py --model_dir models/classic_walk/rf  --motion_key walking --input_dir walk_my_test --out_csv preds_rf.csv
```

## Google Colab examples

[gpu-trained GRU on the running data](https://colab.research.google.com/drive/1FMXT6evpgevoWK_hIyTztyvKdsg9AznN?usp=sharing)