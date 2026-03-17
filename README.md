# StockTrendAI
StockTrendAI: A Stock Market forecasting model
# ⚡ StockTrendAI — Stock Market Forecasting Model

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20GBM%20%7C%20RF-orange?style=flat-square)
![Data](https://img.shields.io/badge/Data-Yahoo%20Finance-purple?style=flat-square)

**An end-to-end AI platform that predicts stock market trends — Up, Down or Sideways — using real-world NVIDIA stock data (2019–2024), combining price forecasting, trend classification and an interactive Streamlit dashboard.**


##  Dashboard Preview

<img width="1890" height="993" alt="Screenshot 2026-03-15 130237" src="https://github.com/user-attachments/assets/fa26ea84-f408-46a8-a0b3-9eaf648298c3" />
<img width="1504" height="811" alt="Screenshot 2026-03-15 125625" src="https://github.com/user-attachments/assets/3fbff332-d195-41f3-85df-802200c1e6c9" />
<img width="1912" height="998" alt="Screenshot 2026-03-15 125643" src="https://github.com/user-attachments/assets/7c91c2a0-5138-46cb-81cc-33d21592f363" />
<img width="1501" height="994" alt="Screenshot 2026-03-15 125706" src="https://github.com/user-attachments/assets/a11470b7-ea20-4b81-aa2c-eff5ec774761" />
<img width="1915" height="991" alt="Screenshot 2026-03-15 125732" src="https://github.com/user-attachments/assets/728abba0-ef1a-4db4-95cd-a511e256f245" />
<img width="1500" height="905" alt="Screenshot 2026-03-15 125757" src="https://github.com/user-attachments/assets/bdec3652-4279-4173-8be0-774843e2891c" />
<img width="1499" height="919" alt="Screenshot 2026-03-15 125820" src="https://github.com/user-attachments/assets/196e4287-a57a-4b9e-a8b4-a0871dc6e612" />
<img width="1898" height="992" alt="Screenshot 2026-03-15 125848" src="https://github.com/user-attachments/assets/9454f97d-3f4a-4f48-8856-619af1ba8d29" />
<img width="1490" height="688" alt="Screenshot 2026-03-15 125908" src="https://github.com/user-attachments/assets/3dcc902e-08f4-48c5-89f4-6870bef86dd8" />
<img width="1423" height="750" alt="Screenshot 2026-03-15 125926" src="https://github.com/user-attachments/assets/ec81e262-62e4-493e-a739-adb051e12e6a" />
<img width="1469" height="786" alt="Screenshot 2026-03-15 125941" src="https://github.com/user-attachments/assets/677687b8-d4ea-4466-aa6f-3da7acaa017c" />
<img width="1892" height="994" alt="Screenshot 2026-03-15 130007" src="https://github.com/user-attachments/assets/dd6aed7c-cd3e-4c52-a4f8-97ec96608e01" />
<img width="1452" height="904" alt="Screenshot 2026-03-15 130025" src="https://github.com/user-attachments/assets/997e90fe-1bab-4b4f-a735-0dfac0f9d815" />
<img width="1417" height="679" alt="Screenshot 2026-03-15 130044" src="https://github.com/user-attachments/assets/29b18b10-08a9-4459-acf9-5ac50d559e67" />
<img width="1475" height="829" alt="Screenshot 2026-03-15 130153" src="https://github.com/user-attachments/assets/09c28507-07e6-4314-95eb-8bdca62f3b38" />
<img width="1911" height="999" alt="Screenshot 2026-03-15 125552" src="https://github.com/user-attachments/assets/6c86aed1-949e-40fe-83af-97e8fc798ed2" />



---

## 📌 About The Project

StockTrendAI is an ML-based stock market forecasting system built on real-world **NVIDIA (NVDA) stock data (2019–2024)** that predicts future stock prices and **Up/Down directional trend**. It covers the complete ML pipeline — from data collection and preprocessing, feature engineering using technical indicators **(RSI, Moving Averages, Momentum)**, to training and comparison of **four models** — Linear Regression, Random Forest, Gradient Boosting and XGBoost. Dedicated **trend classifiers with class balancing** significantly improved directional accuracy beyond the naive ~48% baseline, with all results presented via an **interactive Streamlit dashboard.**

---

## ❗ Problem Statement

- Stock markets are **highly volatile** — traders rely on gut feeling, making manual analysis slow, emotional and unreliable for real buy/sell decisions
- Raw stock data contains **noise, missing values and outliers** — while most systems only predict prices, they completely fail to capture the **Up/Down directional trend**  that traders actually need
- Basic trend prediction gives only **~48% accuracy** — close to random guessing, highlighting the need for a dedicated ML-based trend classifier

---

## ✅ Key Features

- **Complete end-to-end ML pipeline** — data collection to final prediction in one platform
- **4 regression models** trained & compared for best price forecasting accuracy
- **Dedicated trend classifiers** — predicts Up/Down direction, not just price
- **Real-world technical indicators** — RSI, Moving Averages, Momentum & Volatility
- **Interactive Streamlit dashboard** — usable by anyone without technical knowledge
- **Scalable** — works with any stock ticker & date range, not just NVIDIA

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Data Collection | Yahoo Finance API (`yfinance`) |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Model Saving | Joblib |

---

## ⚙️ Core Modules

| Module | Description |
|--------|-------------|
| **Data Collection** | Fetches historical stock data via Yahoo Finance API |
| **Preprocessing** | Handles missing values, duplicates & data type conversion |
| **Outlier Handling** | Detects & smooths outliers using Rolling Z-Score |
| **Feature Engineering** | Computes RSI, Moving Averages, Momentum, Volatility & Lag features |
| **Regression** | Trains & compares 4 models for next-day price forecasting |
| **Trend Classification** | Dedicated classifiers for Up/Down directional trend prediction |
| **Forecasting** | Generates 7-day future price forecast using best performing model |
| **Dashboard** | Presents all results via interactive Streamlit dashboard |

---

## 📊 Features Engineered

| Feature | Purpose |
|---------|---------|
| Moving Averages (MA10, MA20, MA50) | Captures short, medium & long-term price trends |
| RSI | Measures momentum & identifies overbought/oversold conditions |
| Momentum | Measures rate of price change over a 5-day window |
| Volatility | Captures market uncertainty using rolling standard deviation |
| Lag Features (Lag1, Lag2, Lag3) | Previous closing prices for sequential pattern detection |
| Daily Returns | Percentage change in closing price to normalize movements |

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Best Classifier** | Gradient Boosting Classifier |
| **Up Days (Test)** | 118 |
| **Down Days (Test)** | 101 |
| **Total Trading Days** | 1,509 |
| **NVDA Total Return (2019–2024)** | 3970.1% |

---

## 📁 Project Structure

```
StockTrendAI/
│
├── 📄 README.md                    
├── 📄 LICENSE                      
├── 📄 requirements.txt                               
│
├── 📁 data/
│   └── Data is fetched via Yahoo Finance API
│
├── 📁 notebooks/
│   └── StockTrendAI.ipynb         
│
├── 📁 app/
│   └── app.py                     
│
├── 📁 models/
│   └── best_model.pkl            
│
└── 📁 assets/
    └── dashboard_preview.png     
```

---

## 🚀 How To Run

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/StockTrendAI.git
cd StockTrendAI
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit dashboard**
```bash
streamlit run app/app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
streamlit
yfinance
matplotlib
seaborn
joblib
```

---

## 🔚 Conclusion

StockTrendAI successfully proves that ML can be effectively applied to real-world stock market forecasting — combining **price prediction, directional trend classification** and an interactive dashboard into one complete and practical solution for financial decision-making. This project is a **complete end-to-end AI platform** that is simple, fast and accessible for everyone — regardless of technical background.

---

---


---
⭐ **If you found this project helpful, please give it a star!** ⭐
