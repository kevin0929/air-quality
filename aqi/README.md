# aqi 訓練資料夾

資料來源：**台灣環境部（EPA）空氣品質指標（AQI）歷史資料**，涵蓋台南市＋鄰近 4 縣市
（嘉義市、嘉義縣、屏東縣、雲林縣、高雄市、臺南市）共 26 站，2016-11 ～ 2026-05
逐時資料。訓練 PM2.5-GNN / DCRNN / AGCRN / GTS / AirPhyNet 五個模型的空氣品質預測 pipeline。

跟 `../KnowAir-V2/` 是完全獨立的第二份資料來源，模型架構相同，用法也相同。

## 目錄結構

```
aqi/
├── data/
│   ├── raw_data/                  環境部逐月原始 CSV（不進版控，見下方「1. 資料」）
│   ├── build_southwest.py         raw_data/*.csv → combined_southwest.csv + stations_southwest.csv
│   ├── combined_southwest.csv     合併後長格式資料（不進版控，太大）
│   ├── stations_southwest.csv     26 站 metadata（小檔，有進版控）
│   ├── processed/                 preprocess.py 預設輸出（不進版控）
│   └── processed_southwest/       preprocess.py --input combined_southwest.csv 輸出（不進版控）
├── preprocess.py                  共用前處理（見下方「2. 資料前處理」）
├── airphynet/preprocess.py        AirPhyNet 專用前處理入口（呼叫同一份 preprocess.py）
├── models/                        pm25gnn.py / airphynet.py / stgnp.py 模型定義
├── dcrnn_faithful.py / agcrn_faithful.py / gts_faithful.py / airphynet_faithful.py
├── train.py                       統一訓練腳本（pm25gnn / airphynet / stgnp，見下方）
├── train_dcrnn.py / train_agcrn.py / train_gts.py / train_airphynet.py / train_pm25gnn.py
├── test_airphynet.py              載入 checkpoint 在 test split 上評估
├── per_station_mae.py             算每個測站各自的 MAE/RMSE（不是全站平均）
├── data_output.py                 快速攤平 features_raw.npy 成表格檢視用
├── weight/weight_huber_loss.py    WeightedHuberLoss（train.py 使用）
├── utils/                         dataset.py, metrics.py
└── checkpoints/                   訓練好的權重（見下方「4. Model Weight」）
```

## 1. 資料

`data/raw_data/` 放環境部逐月 AQI 歷史資料 CSV（檔名如
`空氣品質指標(AQI)(歷史資料) (2018-08).csv`），可從
[環境部資料開放平台](https://data.moenv.gov.tw/) 下載對應測站、對應月份的歷史資料。
這批原始檔案合計約 950MB，`.gitignore` 已排除，需要你自己下載後放進
`data/raw_data/`。

`data/stations_southwest.csv` 是已經整理好的 26 站 metadata（測站名稱、縣市、經緯度），
已經在 repo 裡，不需重新產生。

## 2. 資料前處理

**Step 1：合併月資料 + 篩選測站**（`data/raw_data/*.csv` → `combined_southwest.csv`）

```bash
python data/build_southwest.py                 # 預設篩選涵蓋率 ≥ 0.85 的測站
python data/build_southwest.py --coverage 0.85
```

**Step 2：轉成訓練用張量**（`combined_southwest.csv` → `features.npy` 等）

```bash
python preprocess.py
```

輸出（`data/processed/` 或 `data/processed_southwest/`，依 `preprocess.py` 內設定）：

| 檔案 | 說明 |
|---|---|
| `features.npy` | `[T, N, F]`，Z-score 正規化後 |
| `features_raw.npy` | `[T, N, F]`，物理單位（PM2.5 為 µg/m³，風速為 m/s）|
| `adj_mx.npy` | `[N, N]` 高斯核地理鄰接矩陣 |
| `adj_wind.npy` | `[N, N]` 考慮風向的有向鄰接矩陣（PM2.5-GNN 用）|
| `dist_mx.npy` | `[N, N]` 站間 Haversine 距離（km）|
| `stations.csv` / `timestamps.csv` | 站點與時間戳 metadata |
| `scalers.pkl` | 正規化用的 mean/std，供反轉換 |
| `split.pkl` | train/val/test 的時間區間 index |

特徵順序（F=10）：`pm2.5, pm10, so2, co, o3, no2, nox, no, wind_u, wind_v`。

AirPhyNet 若要用獨立的 `aqi_all.csv` 走同一套前處理，用 `airphynet/preprocess.py`
（實際邏輯與 `preprocess.py` 相同，只是換輸入/輸出路徑）。

## 3. Training Log

**目前 `aqi/` 底下的訓練腳本沒有接 TensorBoard 或 W&B**（不像
`../KnowAir-V2/train_dcrnn_demo.py`），現有的 checkpoint 是在沒有記錄曲線的情況下
訓練出來的。訓練過程只會印出每個 epoch 的 loss/MAE 到終端機。

若要之後補上可視化訓練紀錄，比照 KnowAir-V2 的做法在對應 `train_*.py` 裡加
`from torch.utils.tensorboard import SummaryWriter`，每個 epoch `writer.add_scalar(...)`
即可，訓練邏輯不需要改動。

## 4. Model Weight

`checkpoints/` 內為各模型訓練好的最佳權重（`.pt`，含 `state_dict` + 訓練時的 `args`）：

| 檔案 | 對應模型 |
|---|---|
| `pm25gnn_best.pt` / `pm25gnn_faithful_best.pt` | PM2.5-GNN |
| `dcrnn_faithful_best.pt` | DCRNN |
| `agcrn_faithful_best.pt` | AGCRN |
| `gts_faithful_best.pt` | GTS |
| `airphynet_faithful_best.pt` | AirPhyNet |

評估：

```bash
python test_airphynet.py --checkpoint checkpoints/airphynet_faithful_best.pt
python per_station_mae.py --model all      # 五個模型都跑，輸出每站 MAE/RMSE
```

## 5. 依賴套件

見專案根目錄 `../requirements.txt`（`pip install -r ../requirements.txt`）。
`aqi/` 這邊實際會用到：`torch`、`numpy`、`pandas`、`tqdm`、`torchdiffeq`（AirPhyNet）。

## 6. 訓練模型

```bash
# PM2.5-GNN / AirPhyNet / STGNP（統一入口）
python train.py --model pm25gnn
python train.py --model pm25gnn --lr 3e-4 --use_season
python train.py --model airphynet --hidden_dim 128 --epochs 150
python train.py --model stgnp --pred_len 48 --batch_size 16

# DCRNN / AGCRN / GTS / AirPhyNet（各自獨立腳本，faithful 復現版本）
python train_dcrnn.py
python train_agcrn.py --embed_dim 10 --hidden_dim 64 --num_layers 2
python train_gts.py --temperature 0.5 --epochs 100
python train_airphynet.py --latent_dim 8 --n_traj 3 --pred_len 24

# PM2.5-GNN 獨立腳本
python train_pm25gnn.py --hist_len 48 --pred_len 24 --epochs 100
```
