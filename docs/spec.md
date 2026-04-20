# 系統規格 v0.1

## 1. 產品定位

目標系統：

- 台股短線 3-10 天選股
- 寶藏股欄位擴展
- 持股追蹤
- Discord 定時推播
- Web UI 管理
- GitHub / Zeabur 部署

## 2. 核心設計原則

1. 不捏造資料
2. 不假設第三方 API 欄位
3. 無真資料不產生結論
4. 所有結果可追溯
5. 所有模組可替換
6. 缺資料要顯性標記
7. 衝突資料保留原文與衝突狀態

## 3. 模組分層

### 3.1 adapters

職責：

- 與資料來源交握
- 驗證欄位
- 轉成內部標準格式
- 保存原始 payload

### 3.2 services

職責：

- 篩選規則
- 分數計算
- 持股監控
- Discord 訊息產生
- 排程協調

### 3.3 routers

職責：

- 提供 API 與 Web 頁面
- 不承擔策略邏輯

## 4. 資料模型

### 4.1 `price_bars`

必要欄位：

- 股票代號
- 交易日
- OHLC
- 成交量
- 資料來源名稱
- 資料來源 URL
- 抓取時間
- 原始 payload

### 4.2 `screening_candidates`

保存：

- 股票代號
- 總分
- 子分數
- evidence
- 進場區
- 止損
- 止盈
- 風報比
- 寶藏股欄位
- 狀態

### 4.3 `holdings`

保存：

- 手動建立持股
- 成本、數量、建立時間
- 備註
- 最新追蹤結果

## 5. 策略規格

### 5.1 短線 3-10 天選股

第一版僅依據真實日線價格運作。

必要前提：

- 至少有 `MIN_PRICE_BARS_FOR_SCREENING` 根真實日線資料

第一版子分數：

- `momentum_5d_score`
- `trend_alignment_score`
- `volume_confirmation_score`
- `risk_reward_score`

### 5.2 寶藏股欄位

第一版先保留欄位與規則入口：

- `treasure_status`
- `treasure_score`
- `treasure_evidence`

若缺營收 / 財報資料，必須標記 `unavailable`。

## 6. Discord 推播規則

排程：

- 週一到週五
- 08:00 Asia/Taipei

若沒有 `ready` 狀態候選股，不發送虛構名單。
