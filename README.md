# 📊 LTV & Cohort Analytics Engine

> A production-grade, interactive analytics dashboard for growth marketing & subscription unit economics — built with Python, Pandas, Scikit-learn, and Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🚀 Features

### 📈 Executive Summary & CAC Payback Optimization
- Blended **LTV:CAC ratio** metric cards
- Quantitative **budget reallocation optimizer** — shifts spend from low-efficiency to high-efficiency channels
- Simulates **~14% theoretical marketing efficiency improvement** via heuristic marginal-return shifting

### 📅 Cohort Retention Matrices
- Monthly cohort retention heatmaps (% of users returning each month)
- Toggleable **ARPU (Average Revenue Per Active User)** heatmap per cohort index
- Built with fully vectorized Pandas pivot operations

### 🔮 Predictive LTV Deep-Dive
- **Random Forest Regressor** trained on RFM features (Recency, Frequency, Monetary) + early-month spend signals
- Plots **actual vs. predicted** cumulative LTV trajectory curves
- Scikit-learn Pipeline with `ColumnTransformer` for numeric + categorical preprocessing

### 🛒 Funnel Diagnostics & Anomaly Detection
- Step-by-step **conversion funnel** with drop-off percentages
- **Rolling Z-score anomaly detection** on weekly checkout conversion rates
- Automatically surfaces and alerts on statistically significant drop-outs
- Injected synthetic checkout anomaly (Oct 10–20) to demonstrate diagnostic engine

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy (fully vectorized) |
| Machine Learning | Scikit-learn (RandomForest, Pipeline, ColumnTransformer) |
| Visualization | Plotly (interactive heatmaps, funnels, time-series) |
| Dashboard | Streamlit |
| Statistical Analysis | SciPy, Rolling Z-Score |

---

## 📦 Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ltv-cohort-analytics.git
cd ltv-cohort-analytics

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
ltv_cohort_analytics/
├── app.py                  # Streamlit dashboard (UI layer)
├── analytics.py            # Core quantitative engine (cohorts, LTV, CAC, funnels)
├── data_generator.py       # Realistic synthetic data simulation (50k+ rows)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme & server config
└── README.md
```

---

## 🧠 Analytical Methods

- **Cohort Analysis**: First-purchase month cohorts, monthly retention grids
- **LTV Forecasting**: Supervised ML regression on early behavioral signals (Months 1–3)
- **CAC Optimization**: Softmax-weighted budget reallocation across paid channels based on LTV/CAC ratios
- **Anomaly Detection**: 4-week rolling mean & standard deviation Z-score thresholding on conversion rates

---

*Built as a portfolio demonstration of production analytics engineering.*
