# KnowAir-V2 專案介紹

## 專案目標

本目錄提供以 KnowAir-V2 空氣品質資料訓練和評估圖時序預測模型的獨立 pipeline。資料集原始檔由 [Zenodo](https://doi.org/10.5281/zenodo.15614907) 發布；本專案把其 netCDF 資料轉成統一的 NumPy 格式，並比較 DCRNN、AGCRN、GTS 與 AirPhyNet 四種模型。`finetune/` 是延伸實驗，使用 ERA5 氣象資料與 CNEMC 空氣品質觀測建立較新時段資料，再微調 DCRNN。

主要區域為長江三角洲（YRD）及京津冀及周邊（BTHSA）。每個監測站視為圖上的節點，逐時特徵作為時間序列；模型以歷史觀測預測未來 PM2.5。KnowAir-V2 前處理使用的特徵包含 PM2.5、O3、氣象變數及風場通道，實際欄位與順序以 `feature_names.json` 或前處理輸出的 metadata 為準。

## 為什麼同時建模時間與空間

空氣污染會隨時間累積、擴散，也會在區域間傳輸。單站時間序列無法利用鄰近站的資訊；單純把所有站點混在一起又會忽略站點間的地理關係。圖時序模型將站點關係和逐時變化一併納入，學習從歷史資料到多步預測的映射。

這裡的圖是由站點座標和程式設定建立的空間先驗，不是完整的大氣流場模擬，也不代表圖上的每條邊都是實際污染傳輸路徑。模型效果仍須在明確的驗證與測試切分上衡量。

## 四種模型概念

| 模型 | 空間關係的處理方式 | 時間預測方式 |
|---|---|---|
| DCRNN | 預先建立的圖上做正向與反向隨機漫步擴散卷積 | Seq2Seq encoder-decoder 自迴歸解碼，訓練含 scheduled sampling |
| AGCRN | 從可學習的節點嵌入產生自適應圖與節點專屬卷積參數 | 循環編碼後輸出多步預測 |
| GTS | 從訓練期站點序列學習離散圖結構，再以圖時序網路預測 | 圖學習與預測器共同訓練 |
| AirPhyNet | 結合圖訊息、風場及可訓練的常微分方程動態 | 以 ODE 演化潛在狀態並預測未來 |

程式是依各論文架構改寫到本專案的資料格式；輸入特徵、建圖方式和訓練細節可能與論文原始基準不同。技術細節見[原始碼說明](原始碼說明.md)。

## 資料與實驗流程

```text
Zenodo netCDF + 站點座標
        ↓ preprocess_knowair.py
features.npy、圖矩陣、scaler、時間切分
        ↓ train_*.py
模型 checkpoint
        ↓ test_models.py / test_gts.py
驗證與測試指標

ERA5 + CNEMC（選用延伸實驗）
        ↓ finetune/build_dataset.py 與 prepare_finetune_data.py
對齊後的延伸資料 + 預訓練 DCRNN
        ↓ finetune/finetune_dcrnn.py
微調 checkpoint、報告與 TensorBoard 記錄
```

前處理產物不只保存數值，也定義站點索引、特徵順序、標準化參數和時間切分。這些資訊與 checkpoint 必須配套；形狀相同但順序不同的輸入仍可能有錯誤語義。執行與資料取得步驟請看[使用手冊](使用手冊.md)及 [finetune 使用手冊](finetune/使用手冊.md)。

## 評估指標與解讀

- **MAE**：預測與觀測的平均絕對差，單位為 μg/m³。
- **RMSE**：平方誤差平均後開根號，較重視較大的偏差。
- **MAPE**：相對誤差百分比；當觀測值接近零時可能不穩定。

請在相同測試資料、相同站點與特徵順序、相同預測時距和前處理條件下比較模型。單一切分的指標不能直接推論所有季節、站點或實際部署情境都會有同樣表現。

## 相關文件

- [使用手冊](使用手冊.md)
- [原始碼說明](原始碼說明.md)
- [根目錄 README](README.md)
- [Fine-tune Introduction](finetune/Introduction.md)
- [Fine-tune 使用手冊](finetune/使用手冊.md)
