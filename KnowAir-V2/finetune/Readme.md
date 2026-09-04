# KnowAir-V2 時間延伸資料集 Pipeline

延用 KnowAir-V2 原本的 355 個站點(BTHSA + YRD),幫你把資料延伸到新的時間段,
輸出格式盡量對齊原始 KnowAir 系列(氣象從 ERA5、空氣品質從 CNEMC),讓你可以拿新時段
資料對 DCRNN / AGCRN / GTS / AirPhyNet 做 fine-tune。

**這套 pipeline 沒辦法在 Claude 的沙盒環境裡直接執行**(網路白名單連不到
`cds.climate.copernicus.eu`、`air.cnemc.cn`、`quotsoft.net`),你需要下載這幾個檔案後,
在自己有網路的機器上跑。

---

## Step 0:兩邊的存取權限申請

### (A) ERA5 / Copernicus CDS API Key

1. 到 https://cds.climate.copernicus.eu/ 註冊帳號(免費)。
2. 登入後,到個人頁面 `https://cds.climate.copernicus.eu/profile` 找到你的 **API Token**。
3. 在你的機器上建立 `~/.cdsapirc` 檔案,內容:
   ```
   url: https://cds.climate.copernicus.eu/api
   key: <你的 API Token>
   ```
4. 第一次呼叫 API 前,到資料集頁面
   `https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels`
   手動點一次「Accept Terms」,不然程式呼叫會被拒絕。
5. 安裝套件:`pip install cdsapi`。
6. 注意:ERA5 正式版通常有 2–3 個月的釋出延遲,最近期的資料只能先抓 **ERA5T**
   (near-real-time,之後數值可能被官方微調)。`fetch_era5.py` 裡會自動偵測日期,
   太新的部分會用 ERA5T 資料集下載並在檔名標註,提醒你之後留意校正。

### (B) CNEMC 空氣品質資料

CNEMC 沒有官方公開的批次歷史資料下載或 API,以下是三種實務可行的管道,難度與穩定性由低到高:

| 方式 | 免申請 | 適合情境 | 備註 |
|---|---|---|---|
| quotsoft.net 歷史 CSV 鏡像 | ✅ 不用註冊 | 補歷史時段(2014/05 起) | `fetch_cnemc.py` 預設用這個 |
| 自建爬蟲抓即時發布介面 | ✅ 不用註冊 | 之後要持續累積最新資料 | 只能抓「當下」,抓不到過去 |
| 阿里雲市場授權 API(付費) | ❌ 需開通阿里雲帳號+購買 | 長期維運、要 SLA 保證 | 官方授權,最穩但要付費 |

`fetch_cnemc.py` 預設用 quotsoft.net(免申請),申請步驟其實就是「不用申請」——
直接照網址規則抓 CSV 即可。如果你之後想換成阿里雲的授權 API 或自建即時爬蟲,
`fetch_cnemc.py` 裡有留一個 `--source` 參數可以切換,程式碼裡標了要改的地方。

若要走阿里雲那條路:
1. 註冊/登入阿里雲帳號:https://market.aliyun.com/
2. 搜尋「國控空氣質量實時發布數據」,選你要的方案(依查詢量計費)。
3. 開通後會拿到 `AppCode`,填進 `config.yaml` 的 `cnemc.aliyun_appcode`。

---

## Step 1:準備站點清單(最關鍵,必須手動完成)

**我這邊沒有 KnowAir-V2 完整的 355 個站點編碼/經緯度清單**——這份清單是隨資料集一起打包的,
不在論文本文裡公開列出。你需要:

1. 從 Zenodo 下載 KnowAir-V2 原始資料集:
   https://doi.org/10.5281/zenodo.15614907
2. 資料集裡通常會附站點的 metadata(編碼、名稱、城市、經緯度)。把它整理成
   `stations.csv`,格式參考本資料夾裡的 `stations_template.csv`。
3. 把整理好的 `stations.csv` 放到專案根目錄,程式會讀這個檔案決定要抓哪 355 個站點、
   以及圖要怎麼建。

如果你手上的預訓練模型是自己训练时用的站點清單(可能存在你的 dataloader 設定檔或
`.npy` 檔的 metadata 裡),務必兩邊互相核對,確保站點順序、數量完全一致——DCRNN /
AGCRN 這幾個模型的輸入是照站點順序排的 tensor,順序錯了模型會學到錯的空間關係。

---

## Step 2:設定 `config.yaml`

打開 `config.yaml`,至少要改:
- `date_range.start` / `date_range.end`:你要延伸的時間段
- `paths.stations_csv`:指到你整理好的站點清單
- `cnemc.source`:`quotsoft`(預設)或 `aliyun`

其他參數(氣象變數清單、圖的距離閾值)已經照 PCDCNet 論文描述的預設值填好,
一般不用動,除非你確定你的預訓練模型用的變數集合不一樣。

---

## Step 3:依序執行

```bash
pip install -r requirements.txt

# 1. 抓 ERA5 氣象格點資料
python fetch_era5.py --config config.yaml

# 2. 抓 CNEMC 空氣品質站點資料
python fetch_cnemc.py --config config.yaml

# 3. 內插到站點座標、建圖、補值、組裝成最終資料集
python build_dataset.py --config config.yaml
```

跑完後,`output/` 資料夾會有:
- `dataset.npy`:形狀 `(time, station, feature)`,feature 依序是
  `[PM2.5, O3, t2m, d2m, tp, sp, blh, msdwswrf, u100, v100]`(順序寫在 `output/feature_names.json`)
- `time_index.csv`:每個時間步對應的實際時間戳
- `graph_adj.npy`:355×355 的鄰接矩陣(200km 地理距離閾值)
- `stations_used.csv`:實際成功配對到資料的站點清單

---

## 已知限制,拿去 fine-tune 前務必自己檢查

1. **輸出的確切 `.npy` shape/欄位順序是我依照 PCDCNet 論文描述重建的**,不是 KnowAir-V2
   官方檔案本身的內部格式。你手上如果已經有能載入原始 KnowAir-V2 的 dataloader 程式碼
   (例如 PM2.5-GNN repo 的 `dataset.py`),**務必對照它的 shape、正規化方式、feature 順序**,
   確保新舊時段串接時模型看到的張量結構完全一致。
2. **缺值補值方法是我自己實作的近似**(短缺值用同站點時間線性內插,長缺值用鄰近站點
   空間平均),原始 KnowAir-V2 論文只說「已做缺值補值」,沒有公開確切演算法,
   兩者不一定完全一樣,新舊時段的補值方式不一致會悄悄引入分布偏移。
3. **正規化統計量**:`build_dataset.py` 預設不做正規化,只輸出原始數值——
   fine-tune 時請沿用你預訓練模型當初用的 mean/std,不要用新資料重新計算。
4. **ERA5 網格內插到站點座標**用的是雙線性內插(bilinear),如果原作者用的是最近鄰
   或其他方法,數值上會有系統性差異。
5. **時區**:ERA5 原始時間是 UTC,CNEMC 發布通常是北京時間(UTC+8)。
   `build_dataset.py` 會統一轉成 UTC 對齊,輸出的 `time_index.csv` 有標註時區。