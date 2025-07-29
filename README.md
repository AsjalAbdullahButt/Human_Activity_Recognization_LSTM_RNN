# DeepHAR – Human Activity Recognition with RNN

DeepHAR is a deep learning-based project designed to classify human activities from motion sensor data using a Recurrent Neural Network (RNN). The system is trained and evaluated using time-series sensor data, providing a robust and visual approach to understanding human movement patterns.

---

## 🗃️ Dataset

The system uses sensor-based time-series data to classify multiple human activities. The following files are used:

- `train.csv` – Labeled data for training the model.
- This is a large file and it can't be uploaded on GitHib ,it can be found on Kaggle in Human Activity Recognization datasets.
- `test.csv` – Data for testing the model’s performance.

Each row in the dataset contains:
- Sensor readings over time (e.g., accelerometer, gyroscope)
- Activity labels (e.g., Walking, Running, Sitting)

---

## 🧠 Model Architecture

The model is built using a Recurrent Neural Network (RNN) architecture, suitable for sequence data. Key features include:

- Input layer: Handles time-series sensor data
- RNN Layers: Captures temporal dependencies
- Dense Output Layer: Softmax activation for multi-class classification

Two saved models are included:
- `har_rnn_model.h5` – Initial trained RNN
- `best_rnn_model.h5` – Final model after tuning and validation

---

## 🏋️ Training and Evaluation

The model was trained using categorical cross-entropy loss and evaluated on multiple metrics.

### 📈 Training Artifacts:
- `training_history.csv` and `training_history.png`: Visualizes loss and accuracy across epochs
- `performance_metrics_by_class.png`: Class-wise precision, recall, F1-score
- `classification_metrics.csv`: Numerical metrics for each class

---

## 📊 Visual Results

A set of visualizations are provided for performance interpretation:

| Visualization | Description |
|---------------|-------------|
| `class_distribution.png` | Distribution of classes in the dataset |
| `confusion_matrix.png` | Raw confusion matrix |
| `normalized_confusion_matrix.png` | Normalized confusion matrix |
| `learning_curves.png` | Accuracy/Loss curves during training |
| `pca_visualization.png` | PCA of feature space for 2D visual inspection |
| `roc_curves.png` | ROC curves for multi-class classification |
| `performance_dashboard.png` | Summary view of all key metrics |
| `performance_metrics_by_class.png` | Class-specific precision/recall/F1 |

---

## 📂 Predictions and Reporting

The model predictions and reports include:

- `predictions.csv` – Predicted activity labels for the test set
- `har_prediction_report.txt` – Full evaluation report with insights

---

## 📑 Reference

This work references methods and concepts from:

- `sensors-17-02556-v3.pdf` – Research literature on HAR using sensor data

---

## 🚀 Usage

To run the model:

```bash
python project2.py
