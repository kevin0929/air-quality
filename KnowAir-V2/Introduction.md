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

## 原始訓練資料與結果分析

本節根據 repository 內的站點 metadata、`preprocess_knowair.py`、已保存的 checkpoint 及 72 小時外推評估 JSON 整理。原始 KnowAir-V2 `.nc` 與 `data/processed_*` 數值檔未包含在目前工作目錄，因此本節不推估 PM2.5 分布、原始缺值率或完整訓練樣本數；資料規模依專案 README 所記錄的資料集資訊，模型指標依 checkpoint/report 原始記錄。

### 資料範圍與處理

KnowAir-V2 原始資料涵蓋 **2016–2023 年逐時資料，共 70,128 個時間點**。站點清單包含 YRD **127 站**及 BTHSA **228 站**，合計 **355 站**。每站輸入 10 個通道：PM2.5、O3、t2m、d2m、sp、tp、blh、msdwswrf、wind_u、wind_v；預測目標是 PM2.5。

`preprocess_knowair.py` 先將時間序列缺值以時間方向線性內插，再以前向/後向填補邊界缺值，最後仍未解決的 NaN 轉為 0。站點地理距離用於建立鄰接矩陣，風場另用於建立風向相關矩陣。資料依時間先後切分：前 70% 為 train、中間 10% 為 validation、最後 20% 為 test；以 70,128 個時間點估算，三段約為 **49,089 / 7,013 / 14,026 個時間點**，實際滑動視窗樣本數會因歷史長度和預測長度而較少。

### YRD 四模型驗證結果

目前保留的 YRD checkpoint 都記錄 24 小時歷史輸入及 24 小時預測，以下列出 checkpoint 內的最佳驗證指標。MAE/RMSE 單位為 μg/m³，MAPE 為百分比。

| 模型 | 最佳 checkpoint epoch | Validation MAE | Validation RMSE | Validation MAPE |
|---|---:|---:|---:|---:|
| DCRNN | 25 | **9.812** | **14.937** | **44.91%** |
| GTS | 21 | 10.162 | 15.379 | 48.93% |
| AGCRN | 22 | 11.124 | 16.510 | 53.09% |
| AirPhyNet | 28 | 11.869 | 17.059 | 62.58% |

在目前保存的這組驗證結果中，DCRNN 的三項指標最低，GTS 次之。這只能說明它們在該次 YRD validation split 上的 checkpoint 表現；目前 repository 沒有同一組設定下四個模型完整的測試集結果，因此不應將此表解讀為最終測試排名或跨區域結論。不同模型的最佳 epoch 也不是訓練成本或泛化能力的直接比較。

另有一個 BTHSA DCRNN checkpoint，記錄的最佳驗證指標為 MAE **20.626**、RMSE **29.824**、MAPE **123.10%**，且最佳 epoch 為 1。此結果明顯高於 YRD checkpoint 的 validation errors，但只有單一 checkpoint 記錄，沒有配套的 BTHSA 模型比較或 test 報告；不能據此判定區域差異成因，也應先核對資料版本、特徵尺度、訓練是否正常停止及對應切分。

### 24 小時模型延伸至 72 小時

`eval72h_dcrnn_faithful_yrd_24h_best_results.json` 記錄一個**以 24 小時預測窗訓練的 YRD DCRNN**，在 test split 上直接自迴歸解碼至 72 小時的結果：整體 MAE **11.658**、RMSE **16.814**、MAPE **73.91%**。分段誤差如下：

| 預測時段 | MAE | RMSE |
|---|---:|---:|
| 第 1–24 小時 | 8.585 | 12.895 |
| 第 25–48 小時 | 12.395 | 17.508 |
| 第 49–72 小時 | 13.992 | 19.374 |

整體誤差隨預測距離增加。逐步 MAE 由第 1 小時約 **3.23** 升至第 24 小時 **10.95**、第 48 小時 **13.22** 和第 72 小時 **14.49 μg/m³**，符合自迴歸預測誤差逐步累積的現象。這項結果是較長 horizon 的外推壓力測試，不能與專門以 72 小時 horizon 訓練的模型視為相同條件比較；本目錄另有 72 小時訓練 checkpoint，但沒有對應的完整評估 JSON 可供並列。

### 解讀限制與後續檢查

- **前處理的標準化有時間資訊洩漏風險。** 目前 `preprocess_knowair.py` 在切分前，以整段資料計算每個特徵的 mean/std，validation 和 test 時段因此參與標準化統計量。這不是標籤洩漏，但在嚴格時間序列評估中，預處理參數應只由 train 時段估計，再套用到 validation/test。已保存的結果反映目前實作，若修改標準化方式，需重新前處理及訓練後才可比較。
- **逐時隨機變化以外的測試情境有限。** 目前採連續時間切分，test 位於資料尾段，適合檢視後段時期表現；但單一切分不足以證明模型在其他年份、季節或其他區域都穩定。
- **需要實際資料才能分析資料品質。** 原始 `.nc` 和 processed arrays 不在本次可檢視檔案中，所以不能由目前的 checkpoint 推導污染物分布、異常值比例、插補比例或各 split 的有效視窗數。
- **指標不能取代誤差診斷。** MAPE 會受接近零的 PM2.5 觀測影響；建議取得資料後補充分站、分季節、預測步長及高污染事件的 MAE/RMSE，並檢查原始極端值和插補資料。

整體而言，已保存的 YRD 驗證 checkpoint 中 DCRNN 指標最佳；其 24 小時模型延伸到 72 小時時誤差明顯隨 horizon 增加。這些結果適合作為目前 pipeline 的基準紀錄，標準化全期間統計和資料檔未隨 repository 提供則是重現及解讀時必須交代的限制。

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
