# Household Electric Power Consumption Forecasting

## Overview

This project develops an end-to-end deep learning pipeline for forecasting household electricity consumption using a stacked Bidirectional Long Short-Term Memory (Bi-LSTM) network. The proposed approach combines advanced data preprocessing, feature engineering, and hyperparameter optimization to improve forecasting accuracy on real-world household energy data.

Household electricity consumption is highly nonlinear and affected by stochastic fluctuations, seasonal patterns, and long-term temporal dependencies. The objective of this project is to predict **Global Active Power** one hour ahead while maintaining high predictive accuracy and efficient inference suitable for practical deployment.

---

# Dataset

**Source:** UCI Individual Household Electric Power Consumption Dataset

**Time Period:** December 2006 – November 2010

**Original Size:** Approximately 2 million minute-level observations

The dataset was resampled to hourly intervals before model development.

### Input Variables

- Global Active Power (target)
- Voltage
- Global Intensity
- Sub Metering 1
- Sub Metering 2
- Sub Metering 3

After feature engineering, the final dataset contains **12 input features**.

---

# Data Preprocessing

A comprehensive preprocessing pipeline was designed to improve data quality and increase the predictive capability of the model.

## Outlier Detection

Outliers were detected using the Interquartile Range (IQR) method. Abnormal values were replaced with missing values and reconstructed through interpolation to preserve temporal continuity.

## Noise Reduction

Exponential Weighted Moving Average (EWMA) smoothing was applied with a smoothing factor of **0.9** to suppress high-frequency fluctuations while retaining long-term trends.

## Feature Engineering

### Lag Features

Historical consumption values were incorporated to capture temporal dependencies.

- Previous 1 hour
- Previous 24 hours
- Previous 168 hours (one week)

### Temporal Features

Additional calendar-based features were introduced.

- Hour of day
- Day of week
- Month

## Feature Scaling

Both input features and target values were normalized using Min-Max Scaling before training.

## Sliding Window

Each training sample consists of:

- Input: 168 consecutive hourly observations
- Output: Global Active Power one hour ahead

**Input Shape**

```
(168, 12)
```

---

# Model Architecture

The forecasting model consists of two stacked Bidirectional LSTM layers followed by a fully connected output layer.

```text
Input (168 × 12)
        │
        ▼
Bidirectional LSTM (128 units)
(return_sequences=True)
        │
        ▼
Dropout (0.1)
        │
        ▼
Bidirectional LSTM (64 units)
        │
        ▼
Dropout (0.1)
        │
        ▼
Dense (1)
```

The network contains approximately **1.2 million trainable parameters**.

Bidirectional LSTM was selected because it learns richer temporal representations than a standard LSTM and helps reduce prediction lag.

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adamax |
| Learning Rate | 0.002 |
| Batch Size | 32 |
| Epochs | 80 |
| Early Stopping | Patience = 15 |

---

# Hyperparameter Optimization

Model hyperparameters were optimized using **Optuna** with Bayesian Optimization.

| Setting | Value |
|---------|------|
| Optimization Library | Optuna |
| Number of Trials | 150 |
| Search Method | Bayesian Optimization |

---

# Evaluation Metrics

Model performance was evaluated using the following regression metrics.

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

---

# Experimental Results

## Performance Comparison

| Metric | Baseline LSTM | Proposed Model | Improvement |
|---------|--------------:|---------------:|------------:|
| R² | 0.5330 | 0.6688 | +25.5% |
| RMSE | 0.4976 kW | 0.3877 kW | -22.1% |
| MAE | 0.3545 kW | 0.2738 kW | -22.7% |

The proposed stacked Bidirectional LSTM consistently outperformed the baseline LSTM across all evaluation metrics.

---

# Ablation Study

The contribution of each component was evaluated through an ablation study.

| Configuration | R² |
|--------------|---:|
| Baseline LSTM | 0.5330 |
| + EWMA | 0.5621 |
| + Lag Features | 0.5987 |
| + Feature Engineering | 0.6214 |
| + Stacked Bidirectional LSTM | 0.6688 |

The results show that every stage of the proposed pipeline contributes to improving forecasting performance.

---

# Practical Evaluation

To assess practical usefulness, prediction errors were translated into estimated electricity billing differences.

- Total prediction error: **0.86%**
- Estimated billing difference: **approximately 175,000 VND**

These results demonstrate that the model is suitable for applications such as:

- Smart homes
- Residential energy management
- Demand response systems
- Energy consumption optimization

---

# Validation Strategy

To prevent data leakage, the following validation strategy was adopted.

- TimeSeriesSplit
- Walk-forward validation
- Purged cross-validation with a 24-hour gap

---

# Deployment Considerations

| Metric | Value |
|---------|------|
| Inference Time | 1.2 ms/sample |
| Model Size | ~1.18M parameters |
| MAPE | < 1% |

The model achieves low-latency inference while maintaining high forecasting accuracy, making it suitable for real-time deployment.

---

# Key Findings

The main observations from this project include:

- Feature engineering provides significant improvements before increasing model complexity.
- Noise reduction enhances learning stability for household energy data.
- Bidirectional LSTM captures temporal dependencies more effectively than a conventional LSTM.
- Combining domain knowledge with deep learning leads to superior forecasting performance.

---

# Future Work

Potential extensions of this work include:

- Multi-step forecasting (24-hour and 7-day horizons)
- Incorporating weather and occupancy information
- Exploring Transformer-based forecasting models such as the Temporal Fusion Transformer
- Developing hybrid architectures combining CNN, LSTM, and attention mechanisms

---

# References

- UCI Individual Household Electric Power Consumption Dataset
- Recent studies on deep learning for energy forecasting
- Research on signal processing techniques including EWMA and IQR-based outlier detection
