# training

PM2.5 / 空氣品質時空預測模型的訓練專案，兩份完全獨立的資料來源、共用同一套模型架構
（PM2.5-GNN、DCRNN、AGCRN、GTS、AirPhyNet）：

| 子資料夾 | 資料來源 | 說明 |
|---|---|---|
| [`aqi/`](aqi/README.md) | 台灣環境部（EPA）AQI 歷史資料 | 台南＋鄰近 4 縣市，26 站，2016-2026 逐時 |
| [`KnowAir-V2/`](KnowAir-V2/README.md) | [KnowAir-V2](https://doi.org/10.5281/zenodo.15614907)（Zenodo） | 長江三角洲／京津冀，各數百站；另含 `finetune/` ERA5+CNEMC 真實資料 fine-tune pipeline |

各自的資料下載方式、前處理步驟、訓練指令、checkpoint 清單，見各子資料夾的 README：
**[`aqi/README.md`](aqi/README.md)** ／ **[`KnowAir-V2/README.md`](KnowAir-V2/README.md)**
（`KnowAir-V2/finetune/` 另有專屬的 **[`finetune/Readme.md`](KnowAir-V2/finetune/Readme.md)**）。

## 依賴套件

```bash
pip install -r requirements.txt
```

`requirements.txt` 涵蓋兩個子資料夾實際用到的套件（PyTorch、xarray/netCDF4、
torchdiffeq、TensorBoard 等）。細節見檔案內註解。

## Training Log

只有 `KnowAir-V2/`（DCRNN demo + fine-tune）有實際跑出 TensorBoard log，格式為標準
TensorBoard event files，可直接用 TensorBoard GUI 開啟：

```bash
pip install tensorboard
tensorboard --logdir KnowAir-V2/runs
tensorboard --logdir KnowAir-V2/finetune/runs_finetune
```

`aqi/` 目前沒有訓練紀錄（腳本沒有接 TensorBoard/W&B），細節見
[`aqi/README.md`](aqi/README.md#3-training-log)。

## Model Weight

兩個子資料夾的 `checkpoints*/` 都已經包含訓練好的 `.pt` 權重，清單見各自 README。

## 資料

原始/前處理後資料檔案太大（合計數 GB），沒有放進這個 git repo（見 `.gitignore`），
各子資料夾 README 有寫清楚要去哪裡下載、怎麼重新跑前處理產生。
