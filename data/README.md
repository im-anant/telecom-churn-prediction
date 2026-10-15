# Dataset

The raw CSV dataset is **not included** in this repository to keep it lightweight.

## How to get the data

### Option A — Automatic download (recommended)
The notebook downloads the dataset automatically using `kagglehub` when you run the first cell.  
You need a Kaggle account and your `kaggle.json` API key placed at `~/.kaggle/kaggle.json`.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
**Kaggle slug:** `blastchar/telco-customer-churn`

### Option B — Manual download
1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Click **Download** and unzip the file.
3. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside this `data/` folder.
4. The notebook will automatically detect and use the local file if the Kaggle download fails.

## File details
| File | Rows | Columns | Target |
|---|---|---|---|
| WA_Fn-UseC_-Telco-Customer-Churn.csv | 7,043 | 21 | `Churn` (Yes/No) |
