
# ⚡ Household Electric Power Forecasting
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 🚀 Project Overview
An advanced **time series forecasting system** for predicting household electricity consumption using:

- 🔥 Stacked Bidirectional LSTM
- 🧠 Advanced Feature Engineering
- 🧹 Signal Processing (EWMA + IQR)
- ⚙️ Optuna Hyperparameter Optimization

👉 Goal: Predict **electric power consumption (kW)** 1 hour ahead with high accuracy.

---

## 🎬 Demo (Placeholder)
![Demo GIF](https://via.placeholder.com/800x400.png?text=Demo+GIF+Here)

---

## 📊 Results

| Metric | Baseline | Proposed | Improvement |
|--------|---------|----------|------------|
| R²     | 0.5330  | 0.6688   | +25.5% |
| RMSE   | 0.4976  | 0.3877   | -22.1% |
| MAE    | 0.3545  | 0.2738   | -22.7% |

---

## 🧠 Model Architecture

```
Input (168 × 12)
↓
BiLSTM (128)
↓
Dropout
↓
BiLSTM (64)
↓
Dense
```

---

## 🔧 Pipeline

### Data Processing
- IQR Outlier Removal
- EWMA Denoising
- Lag Features (1h, 24h, 168h)
- Temporal Encoding

### Training
- Sliding Window (168 timesteps)
- TimeSeries CV (no leakage)
- Optuna tuning

---

## 💰 Business Impact

- ⚡ Error: <1% MAPE
- 💸 Billing difference: ~175k VND
- ⏱️ Inference: 1.2ms/sample

---

## 📂 Project Structure

```
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt
```

---

## ⚡ Installation

```bash
git clone https://github.com/your-repo/electric-forecasting.git
cd electric-forecasting
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python train.py
python evaluate.py
```

---

## 🧠 Key Insights

- Feature engineering boosts performance significantly
- Denoising is critical for real-world time series
- BiLSTM reduces phase lag vs standard LSTM

---

## 🔮 Future Work

- Multi-step forecasting
- Transformer models (TFT)
- Weather & external features

---

## 👨‍💻 Author

Data Science Project – Energy Forecasting  
FPT University  

---

## ⭐ If you find this useful, give it a star!
