# KnowAir-V2 訓練資料夾

跟 `../aqi/` 資料夾完全分開的獨立 pipeline：DCRNN / AGCRN / GTS / AirPhyNet 四個模型
沿用一模一樣的架構與訓練邏輯，只是換成 [KnowAir-V2](https://doi.org/10.5281/zenodo.15614907) 資料集。
另外 `finetune/` 底下是拿 ERA5 + CNEMC 真實資料延伸時間段、對 DCRNN 做 fine-tune 的子專案。

## 文件入口

- [Introduction.md](Introduction.md)：專案目的、資料與四種模型的概念導覽。
- [使用手冊.md](使用手冊.md)：環境設定、資料前處理、訓練、評估與 TensorBoard 操作。
- [原始碼說明.md](原始碼說明.md)：根目錄各程式模組及資料流的說明。
- [`finetune/`](finetune/Readme.md)：ERA5/CNEMC 時間延伸資料、DCRNN 微調的完整說明。

## 目錄結構

```
KnowAir-V2/
├── data/                        資料（見下方「1. 資料」）
│   ├── dataset_yrd.nc             原始 KnowAir-V2 檔案（不進版控，見 .gitignore）
│   ├── dataset_bthsa.nc
│   ├── stations_yrd.csv           站點座標（小檔，有進版控）
│   ├── stations_bthsa.csv
│   ├── processed_yrd/             preprocess_knowair.py 產生（不進版控）
│   └── processed_bthsa/
├── preprocess_knowair.py        .nc → features.npy 等 aqi pipeline 相容格式
├── dcrnn_faithful.py             模型架構（跟 aqi/ 版本相同）
├── agcrn_faithful.py
├── gts_faithful.py
├── airphynet_faithful.py
├── train_dcrnn.py                訓練腳本（只改了預設 --data 路徑）
├── train_dcrnn_demo.py           帶 TensorBoard/W&B log 的 DCRNN demo 訓練腳本
├── train_agcrn.py / train_gts.py / train_airphynet.py
├── test_models.py / test_gts.py  只做推論，讀 checkpoint 算 MAE/RMSE/MAPE
├── test_extended_horizon.py      拿 24h checkpoint 直接外推到 72h 評估
├── checkpoints/                  訓練好的權重（見下方「3. Model Weight」）
├── checkpoints_yrd/ ・ checkpoints_bthsa/
├── runs/                          TensorBoard log（見下方「2. Training Log」）
└── finetune/                      ERA5 + CNEMC 真實資料 fine-tune 子專案（見 finetune/Readme.md）
```

## 1. 資料

KnowAir-V2 實際放在 **Zenodo**，不是 GitHub，且 `.nc` 檔案太大（合計約 950MB）
不放進 git repo，`.gitignore` 已排除，需要你自己下載：

> Wang, S., Cheng, Y., Meng, Q., Saukh, O., Zhang, J., Fan, J., Zhang, Y., Yuan, X., & Thiele, L. (2025).
> KnowAir-V2: A Benchmark Dataset for Air Quality Forecasting with PCDCNet [Data set]. Zenodo.
> **https://doi.org/10.5281/zenodo.15614907**

下載後把 4 個檔案放進 `data/`（跟 `data/stations_yrd.csv` 平級，這兩個小的 station csv 已經在 repo 裡）：

```
data/
├── dataset_yrd.nc
├── dataset_bthsa.nc
├── stations_yrd.csv     ← 已在 repo
└── stations_bthsa.csv   ← 已在 repo
```

### 前處理（把 .nc 轉成跟 aqi pipeline 相容的格式）

```bash
python preprocess_knowair.py --region yrd   --raw_dir data
python preprocess_knowair.py --region bthsa --raw_dir data
```

會在 `data/processed_yrd/`、`data/processed_bthsa/` 產生 `features.npy` / `adj_mx.npy` /
`scalers.pkl` / `split.pkl` 等檔案，跟 `aqi/preprocess.py` 輸出的 schema 完全一致。

**第一次執行務必檢查終端機印出的「.nc 檔案內容診斷」區塊**——`VAR_MAP_CANDIDATES`
（變數名稱對照表）是依照 ERA5 / KnowAir 命名慣例推測的，如果印出「⚠ 找不到 xxx 對應的變數」，
對照印出的實際變數清單，回頭修改 `preprocess_knowair.py` 最上方的 `VAR_MAP_CANDIDATES`。

### ERA5 資料前處理（`finetune/` 子專案）

如果要用真實 ERA5（氣象）+ CNEMC（空氣品質）資料延伸時間段來 fine-tune，
完整前處理程式碼在 `finetune/`：

- `finetune/fetch_era5.py` — 從 Copernicus CDS API 下載 ERA5/ERA5T 再分析氣象資料
- `finetune/fetch_cnemc.py` — 下載 CNEMC 空氣品質站點資料
- `finetune/build_dataset.py` — 內插到站點座標、建圖、補值、組裝成最終資料集
- `finetune/prepare_finetune_data.py` / `finetune/weight_interpolation.py` — 銜接到
  `aqi`/`preprocess.py` schema 並做權重內插

完整步驟（CDS API key 申請、CNEMC 資料來源、`config.yaml` 設定）見
**[`finetune/Readme.md`](finetune/Readme.md)**，這邊不重複貼。

## 2. Training Log（TensorBoard）

目前只有 **DCRNN**（demo：`train_dcrnn_demo.py`，以及 `finetune/finetune_dcrnn.py`）
有接 TensorBoard／可選 W&B 的訓練紀錄；AGCRN / GTS / AirPhyNet 的 `train_*.py` 目前
沒有寫 log。已經產生的 log event files在：

```
runs/                              ← train_dcrnn_demo.py 的訓練紀錄
├── yrd_24h/
├── yrd_72h/
└── eval72h_dcrnn_faithful_yrd_24h_best/

finetune/runs_finetune/            ← finetune_dcrnn.py 各次 fine-tune 的訓練紀錄
└── finetune_YYYYMMDD-HHMMSS/
```

本地開啟 TensorBoard GUI：

```bash
pip install tensorboard
tensorboard --logdir runs                      # 看 demo 訓練紀錄
tensorboard --logdir finetune/runs_finetune     # 看 fine-tune 訓練紀錄
```

打開 http://localhost:6006 即可看到 loss / MAE 曲線。

要重新產生（或幫其他模型補上）log，`train_dcrnn_demo.py` 內建：

```bash
python train_dcrnn_demo.py --exp_name my_run
python train_dcrnn_demo.py --wandb --wandb_project dcrnn-knowair   # 選用，另外鏡像到 W&B
```

## 3. Model Weight

已訓練好的 checkpoint（`.pt`，內含 `state_dict` 與訓練時的 `args`，`test_models.py`
會直接從 checkpoint 還原超參數，不需要手動重打）：

| 檔案 | 說明 |
|---|---|
| `checkpoints/dcrnn_faithful_yrd_24h_best.pt` | DCRNN，YRD 區域，24h 預測窗 |
| `checkpoints/dcrnn_faithful_yrd_72h_best.pt` | DCRNN，YRD 區域，72h 預測窗 |
| `checkpoints_yrd/{dcrnn,agcrn,gts,airphynet}_faithful_best.pt` | YRD 區域四個模型 |
| `checkpoints_bthsa/dcrnn_faithful_best.pt` | BTHSA 區域 DCRNN |
| `finetune/checkpoints_finetune/dcrnn_finetuned_finetune_*_best.pt` | 用 ERA5/CNEMC 真實資料 fine-tune 後的 DCRNN（含對應 `finetune_report_*.json` / `baseline_metrics_*.json` 評估報告）|

載入評估：

```bash
python test_models.py --model dcrnn --ckpt checkpoints_yrd/dcrnn_faithful_best.pt
python test_gts.py
python test_extended_horizon.py --ckpt checkpoints/dcrnn_faithful_yrd_24h_best.pt   # 24h 權重外推到 72h
```

## 4. 依賴套件

見專案根目錄 `../requirements.txt`（`pip install -r ../requirements.txt`）。
KnowAir-V2 這邊實際會用到：`torch`、`xarray`、`netCDF4`（讀 `.nc`）、`torchdiffeq`
（只有 AirPhyNet 需要）、`PyYAML` + `requests` + `cdsapi`（`finetune/` 抓 ERA5/CNEMC）、
`matplotlib`（`train_dcrnn_demo.py` / `test_extended_horizon.py` 畫曲線圖）、
`tensorboard`（開 log GUI）。

## 5. 訓練四個模型

跟 `aqi/` 資料夾的用法完全一樣，只是 `--data` 指向這裡的資料夾：

```bash
# 訓練 YRD 區域
python train_dcrnn.py     --data data/processed_yrd
python train_agcrn.py     --data data/processed_yrd
python train_gts.py       --data data/processed_yrd
python train_airphynet.py --data data/processed_yrd

# 訓練 BTHSA 區域
python train_dcrnn.py     --data data/processed_bthsa
python train_agcrn.py     --data data/processed_bthsa
python train_gts.py       --data data/processed_bthsa
python train_airphynet.py --data data/processed_bthsa
```

`--data` 預設值已經設成 `data/processed_yrd`，所以不加這個參數也會跑 YRD。

## 跟 aqi/ 資料夾的差異

| 項目 | aqi/ | KnowAir-V2/ |
|---|---|---|
| 資料來源 | 台灣環境部 AQI API | KnowAir-V2（Zenodo）|
| 涵蓋範圍 | 台南+鄰近4縣市，26站 | 長江三角洲 / 京津冀，各自數百站 |
| 時間跨度 | 2016–2026（逐時）| 2016–2023（逐時，70,128筆）|
| PM2.5-GNN | 有訓練 | 未包含（這裡只做 DCRNN/AGCRN/GTS/AirPhyNet）|
| Training Log | 無 | 有（DCRNN demo + finetune，見上方「2. Training Log」）|
| 模型架構 | 完全相同 | 完全相同 |
