
# LSTM Flare Anomaly Detection

This project uses Long Short-Term Memory (LSTM) neural networks to perform anomaly detection on time series data related to flare activities. The model is trained and evaluated using normalized sensor data and labeled events.

## 📁 Repository Contents

- **LSTM_Flare_Anomaly_Detection.ipynb**  
  Jupyter Notebook implementing an LSTM-based model for anomaly detection. Includes data preprocessing, model training, evaluation, and visualization.

- **data.csv**  
  The original unprocessed dataset containing raw time series data.

- **normalized_testing.csv**  
  Preprocessed and normalized version of the testing dataset used during model evaluation.

- **labels.csv**  
  Contains true class labels for the time series data used to evaluate model performance.

- **classification_report.txt**  
  Model performance summary with metrics such as precision, recall, and F1-score.  
  > ⚠️ Note: Class `1` has zero support, indicating no anomalies were detected in the testing set.

## 🧠 Model Overview

- **Model Type**: LSTM (Long Short-Term Memory)
- **Objective**: Binary classification (Normal vs Anomaly)
- **Metrics Used**: Accuracy, Precision, Recall, F1-score

## 📊 Evaluation Summary

```text
Accuracy:      0.95  
Precision:     1.00 (for class 0)  
Recall:        0.95 (for class 0)  
F1 Score:      0.97 (for class 0)
```

> **Note**: Anomalous class (`1`) had no samples in the test set, which is reflected in zero values across its evaluation metrics.

## 🛠️ How to Run

1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow
   ```

2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook LSTM_Flare_Anomaly_Detection.ipynb
   ```

3. Run cells sequentially to preprocess data, train the model, and view results.

## 📌 Future Work

- Add more diverse labeled anomaly data
- Test model on real-time or streaming data
- Evaluate different architectures (e.g., GRU, Transformer)
