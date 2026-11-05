# Fine-tune 量化結果介紹

## 專案簡介：DCRNN、KnowAir-V2 與 ERA5 fine-tune

### 為什麼要做這個專案

空氣品質預測看起來像是在回答「明天 PM2.5 會是多少」，實際上同時牽涉兩種會互相影響的變化：污染濃度會隨時間累積、擴散與消散；不同監測站之間也會因為地理位置和區域傳輸而呈現關聯。只把每個站點當成互不相干的一條時間序列，會忽略污染可能從鄰近區域移動過來；只看站點間的空間關係，又無法描述一天之中污染如何升降。因此，這個專案以圖神經網路處理站點之間的空間連結，再以循環神經網路處理逐時序列，並透過 fine-tune 把已學到的規律帶到新的資料時段。

這裡的重點不是假設每個地區都永遠遵循同一種污染型態，而是先利用 KnowAir-V2 的長時間、多站點資料訓練一個有基礎能力的預測模型，再用較新的 ERA5 氣象與 CNEMC 空品資料做適應。這樣做可以讓實驗比較「直接沿用預訓練模型」和「經過新時段微調」的表現差異，也能透過測試集指標檢查微調到底有沒有帶來實際改善。

### DCRNN 是什麼

DCRNN（Diffusion Convolutional Recurrent Neural Network）是用於圖結構時序預測的模型。這個專案把空氣品質監測站視為圖上的節點，以站點間的地理鄰接關係描述空間連結；歷史逐時觀測則構成每個節點的時間序列。

模型由兩部分組成：擴散卷積 GRU（DCGRU）沿圖的正向與反向 random walk 傳遞資訊，學習相鄰站點的空間關係；Seq2Seq encoder-decoder 先讀取歷史時段，再自回歸地逐步預測未來。可以把 encoder 想成先整理過去一段時間的狀態摘要，decoder 則從這份摘要出發，一小步一小步產生未來預測。因為每一步的輸出會成為下一步預測的輸入，前面的小誤差可能沿預測時間累積，所以訓練時使用 scheduled sampling，逐漸降低 decoder 使用真實前一步答案的機率，讓訓練情境更接近實際推論時只能使用模型自己預測值的情況。此專案以 PM2.5 作為預測目標，其他特徵可作為輸入。

「擴散」在這裡不是指模型直接模擬完整的大氣物理方程，而是指沿著圖上的隨機漫步轉移矩陣傳遞與彙整鄰站資訊。圖卷積會參考不同擴散步數的訊息，讓模型同時接觸直接相鄰站點及較遠一層的空間訊號。站點圖提供模型一個明確的空間骨架，GRU 則負責從連續觀測中學習變化的先後關係；兩者結合後，模型可以一起利用「附近站點現在如何」和「本站最近如何變化」這兩類線索。

當然，圖上的連結本身不等於真實風場或污染傳輸路徑，也不會自動保證每個鄰站都對預測有幫助。它是一種依照資料與建圖方式提供給模型的空間先驗，模型再從訓練資料中學習如何使用。這也是為什麼資料的站點順序、鄰接矩陣和訓練時版本必須保持一致：模型權重學到的空間關係，依賴輸入中每一個節點所代表的站點。

### KnowAir-V2 預訓練

`KnowAir-V2/` 提供 DCRNN 等模型在 KnowAir-V2 資料上的訓練流程。資料集涵蓋長江三角洲（YRD）與京津冀（BTHSA）監測站時序；下載原始資料後，由 `preprocess_knowair.py` 轉為模型使用的 `features.npy`、`adj_mx.npy`、`scalers.pkl` 和 `split.pkl`。這些檔案分別承載模型輸入特徵、站點圖、標準化統計量和資料切分資訊，讓訓練程式能使用一致的格式載入資料。

`train_dcrnn.py` 讀取這些產物，使用訓練切分更新模型，並依驗證表現保存最佳 checkpoint。訓練資料提供模型學習參數的例子，驗證資料協助監控訓練進度及挑選 checkpoint，測試資料則用來估計模型對未參與訓練資料的表現。預訓練的價值在於，模型不必從隨機權重開始摸索所有規律，而能以已學到的空間與時間模式作為新實驗的起點。它並不代表新地區或新時段一定會表現良好，仍須透過對應資料評估。

### ERA5/CNEMC 延伸資料與 fine-tune

`KnowAir-V2/finetune/` 子專案將訓練時段延伸到新日期：`fetch_era5.py` 下載 ERA5/ERA5T 再分析氣象資料，`fetch_cnemc.py` 取得 CNEMC 空氣品質觀測，`build_dataset.py` 對齊時間與站點、處理缺值並輸出資料。ERA5 是氣象資料來源；空氣品質目標則來自 CNEMC，並非 ERA5 直接提供 PM2.5。兩種來源扮演不同角色：ERA5 描述同一時段的大尺度氣象環境，CNEMC 提供地面監測到的污染物觀測，模型再以歷史觀測和氣象特徵預測 PM2.5。

這段資料整理工作看似只是把檔案湊在一起，卻是整個 fine-tune 能否成立的關鍵。來源資料要先使用共同的時間基準，站點資料要能對應到模型預期的節點，特徵欄位也要使用訓練時一致的語義和排列。缺值處理、時間對齊、空間圖建立與標準化方式若有落差，都可能讓模型接收到分布不同或意義錯置的輸入；即使程式順利執行，也不能因此推論資料已完全可比。因此，pipeline 會將這些步驟明確拆開，讓原始下載、資料建置和模型訓練各自留下可檢查的產物。

接著 `prepare_finetune_data.py` 依原預訓練資料的 feature 順序和 scaler 標準化延伸資料，建立鄰接矩陣與 train/validation/test 切分；`finetune_dcrnn.py` 載入 KnowAir-V2 預訓練 DCRNN，只微調模型權重，並輸出 checkpoint、量化報告與 TensorBoard event logs。微調通常使用比完整預訓練更小的學習率和較短訓練週期，目的在保留原模型已學到的能力，同時讓參數適應新的資料分布。實際訓練仍會依驗證結果挑選模型；最後是否有改善，要看在相同測試資料、相同前處理和相同評估方式下的量化比較。

延伸資料的站點、特徵順序及 scaler 必須與模型相容，否則即使張量維度相同，模型輸入的語意也可能錯置。比如模型預期某個 channel 是 PM2.5，卻收到另一個氣象變數，形狀上完全合法，結果仍然沒有意義。`prepare_finetune_data.py` 因此會依原始 scaler 中記錄的 feature 名稱重排資料，並沿用原始 mean/std，而不是對新資料重新計算標準化統計量，減少新舊資料的尺度定義不一致。

整體流程可概括為：

```text
KnowAir-V2 原始資料 → 前處理 → DCRNN 預訓練 checkpoint
ERA5 氣象 + CNEMC 空品 → 延伸資料建置 → 對齊原 feature/scaler
預訓練 checkpoint + 延伸資料 → DCRNN fine-tune → 權重、評估報告、TensorBoard logs
```

## 訓練資料與實驗分析

以下分析根據本目錄目前保留的 `output/` 資料產物，以及 `checkpoints_finetune/` 中的 baseline 與 fine-tune JSON 報告整理；沒有重新訓練模型。資料概況代表交付的建置輸出，模型結果則是各 JSON 所記錄的測試集指標。

### 延伸資料概況

目前 `output/dataset.npy` 的形狀為 **(8,784 小時, 126 站, 10 特徵)**，時間涵蓋 2024-01-01 至 2024-12-31，站點皆屬 YRD。特徵依序為 PM2.5、O3、t2m、d2m、tp、sp、blh、msdwswrf、u100、v100。換言之，這份已交付的訓練資料是 YRD 的 126 站資料，不是把 YRD 與 BTHSA 合併後的 355 站資料；解讀 checkpoint 時應以實際配套資料為準。

`station_completeness.csv` 記錄的原始缺值比例顯示，126 站 PM2.5 平均缺值約 **3.16%**，O3 約 **3.61%**，氣象特徵在該表中沒有缺值；但個別站的 PM2.5 與 O3 缺值最高分別約 **72.52%**、**72.20%**。建置後的 `dataset.npy` 沒有 NaN，表示缺值已經過補值，並不代表原始觀測完整。個別高缺值站的時間序列可能有相當比例來自插補，建議後續報告另做排除高缺值站的敏感度分析。

在目前輸出的 PM2.5 值中位數為 22、平均數約 28.86，最大值為 1,330（依資料原單位）。最大值遠高於中位數，代表分布有長尾或極端值；它可能反映真實污染尖峰，也可能涉及異常觀測或合併資料的品質問題。僅憑摘要統計無法判定原因，故在解讀 RMSE 前應檢視極端值的時間、站點和來源。MAPE 亦可能因低濃度觀測而放大，不能單獨作為模型優劣依據。

### 比較方法

各 JSON 報告先在同一延伸資料切分上計算預訓練模型 baseline，再記錄 fine-tune 後的測試指標。以下列出能代表不同起始 checkpoint 的結果；百分比改善沿用報告定義，以 `(微調前誤差 - 微調後誤差) / 微調前誤差` 計算，正值代表誤差下降。所有 MAE/RMSE 數值單位沿用原訓練報告（μg/m³）。

| 實驗報告 | 預訓練 checkpoint | 測試 MAE（前 → 後） | 測試 RMSE（前 → 後） | 測試 MAPE（前 → 後） | 報告結果 |
|---|---|---:|---:|---:|---|
| `finetune_report_ft_from_5ep_v5_clean.json` | `dcrnn_faithful_yrd_5ep_best.pt` | 10.385 → 9.236（改善 11.06%） | 13.805 → 13.443（改善 2.62%） | 82.26% → 59.56% | `accepted: true` |
| `finetune_report_ft_from_10ep_v5_clean.json` | `dcrnn_faithful_yrd_10ep_best.pt` | 8.548 → 8.805（退步 3.00%） | 12.378 → 12.850（退步 3.81%） | 50.78% → 54.20% | `accepted: false` |
| `finetune_report_finetune_20260903-005712.json` | `dcrnn_faithful_best.pt` | 8.133 → 8.072（改善 0.75%） | 11.908 → 11.822（改善 0.72%） | 48.73% → 48.29% | 未標記 accepted |

### 結果解讀

1. **微調沒有一致的增益。** 五 epoch checkpoint 的 v5 clean 實驗三項指標均下降；十 epoch checkpoint 的 v5 clean 實驗三項指標均上升。另有多次微調未找到優於 baseline 的 checkpoint，或 MAE/RMSE 變差。微調成效與起始權重及訓練設定有關，不能概括成「fine-tune 一定提升」。
2. **改善幅度與絕對表現是兩件事。** 五 epoch 實驗相對自身 baseline 的改善最大，但微調後 RMSE 仍為 13.443；較強的 `dcrnn_faithful_best.pt` baseline 為 11.908，該系列微調後為 11.822。前者不能因相對改善百分比更高，就判為整體預測表現最佳。跨不同預訓練 checkpoint 的數值也不應視為完全等條件比較。
3. **較強 baseline 的增益幅度有限。** `finetune_20260903-005712` 的 MAE/RMSE 相對改善不到 1%，MAPE 由 48.73% 降至 48.29%。這是正向但幅度小的單一切分結果，現有報告沒有多隨機種子信賴區間或顯著性檢定，尚不能判定改善是否穩健。
4. **MAPE 與 MAE/RMSE 的解讀可能不一致。** 五 epoch v5 clean 的 MAPE 大幅下降，而 RMSE 僅改善 2.62%；MAPE 對低 PM2.5 觀測特別敏感，因此應連同 MAE、RMSE 及分站/分時段誤差一起閱讀。對尖峰的影響則應由分位數或極端事件分析補充。

### 切分與研究限制

`prepare_finetune_data.py` 預設採逐月分層切分：每個月內依序切出 train、validation、test（預設比例 70%/10%/20%），讓三個集合都涵蓋一年中的各月份，並避免滑動視窗跨越切分區段。這可降低季節分布不均造成的比較偏差，但 test 並非一段完全晚於 train 的未來時間區間，因此此結果不能直接代表嚴格的未來年度外推能力。

另有一項重要評估限制：`finetune_report_ft_from_5ep_v5_clean.json` 記錄的 `acceptance_metric` 是 `test-RMSE`，而且保留多次實驗結果。若依測試集表現挑選或接受 checkpoint，該測試集就參與了模型選擇，不能再當成完全獨立的最終測試集。因此 `accepted: true` 是該報告流程內的接受標記，不等同外部驗證或統計顯著性證據。較嚴謹的後續流程應只用 validation 選 checkpoint/超參數，鎖定模型後再於未參與決策的時間段做一次最終測試。

本分析還受限於：

- `output/` 僅提供一個 2024 年 YRD 建置資料集，未涵蓋其他年度或 BTHSA 的結果；原始缺值補值策略可能影響各站資料分布。
- 訓練報告使用的 `data/processed_yrd_finetune` 沒有隨目前目錄附上，無法從交付檔案重建每次訓練切分並逐項核對。
- 各報告保存的細節與欄位不完全一致；表格只整理共同可比的測試指標，不以缺少資料的欄位推測結果。
- 本次只檢視既有輸出與報告，未重新訓練、重算指標或確認所有 checkpoint 與報告的二進位內容一致。

因此，目前較穩健的結論是：**在已記錄的 2024 YRD 延伸資料實驗中，fine-tune 有時能改善對應 baseline，但增益高度依賴起始 checkpoint；現有資料不足以支持普遍提升的結論。**

## 建議引用的代表性 checkpoint

若要重現報告中標記接受的實驗，可使用：

```text
checkpoints_finetune/dcrnn_finetuned_ft_from_5ep_v5_clean_best.pt
```

其報告為 `checkpoints_finetune/finetune_report_ft_from_5ep_v5_clean.json`，比較結果見上表。若目標是查看較強 baseline 起始權重的微調結果，則可參考 `dcrnn_finetuned_finetune_20260903-005712_best.pt` 及同名 JSON 報告；該組的測試 RMSE 改善約 0.72%，但未標示為接受模型。無論選用哪一個 checkpoint，都必須配套使用訓練時一致的資料版本、站點順序、特徵順序與 scaler。

## TensorBoard

本資料夾已保留 `runs_finetune/` 下的 TensorBoard event files，包含 `ft_from_5ep_v5_clean` 等實驗。安裝根目錄 `requirements.txt` 的套件後，在此資料夾執行：

```bash
tensorboard --logdir runs_finetune --port 6006
```

瀏覽器開啟 `http://localhost:6006`，在 Scalars 面板查看訓練曲線。若要只查看最新實驗：

```bash
tensorboard --logdir runs_finetune/ft_from_5ep_v5_clean --port 6006
```
