# 🚗 Sensor-Based Driving Behavior Classification

This project aims to classify driver behaviors using machine learning techniques based on vehicle telemetry data obtained from accelerometer and gyroscope sensors.

---

## 📌 Problem Description

Understanding driver behavior is critical for:

* Advanced Driver Assistance Systems (ADAS)
* Intelligent Transportation Systems (ITS)
* Road safety improvement

In this project, risky driving behaviors are classified using statistical features extracted from sensor data.

---

## 📊 Dataset

The dataset is obtained from Kaggle:

👉 [Driving Behavior Dataset](https://www.kaggle.com/datasets/shashwatwork/driving-behavior-dataset/data?utm_source=chatgpt.com)

### Features

The dataset contains **statistical features** derived from raw sensor signals:

* Mean
* Standard Deviation
* Minimum / Maximum
* Skewness
* Kurtosis
* Variance
* Covariance

Sensors used:

* Accelerometer (AccX, AccY, AccZ)
* Gyroscope (GyroX, GyroY, GyroZ)

---

## 🎯 Target Classes

The classification task includes four driving behaviors:

| Class | Description         |
| ----- | ------------------- |
| 1     | Sudden Acceleration |
| 2     | Sudden Right Turn   |
| 3     | Sudden Left Turn    |
| 4     | Sudden Braking      |

---

## ⚙️ Project Pipeline

The project follows a structured machine learning pipeline:

1. **Data Analysis (EDA)**

   * Dataset inspection
   * Class distribution visualization
   * Correlation analysis

2. **Data Preprocessing**

   * Train-test split (80% - 20%)
   * Stratified sampling
   * Feature scaling (StandardScaler)

3. **Model Training & Evaluation**

   * K-Nearest Neighbors (KNN)
   * Logistic Regression
   * Support Vector Machine (SVM)
   * Decision Tree
   * Random Forest

4. **Model Evaluation Metrics**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix

---

## 🔍 Key Findings

* All models achieved high performance (~99% accuracy)
* Random Forest achieved perfect accuracy on the test set
* Cross-validation confirmed strong generalization (~99%)

📌 This indicates that the extracted statistical features provide strong class separability.


---

## 📁 Project Structure

```
DrivingBehaviorClassification/
│
├── dataset/
├── outputs/
│   ├── knn/
│   ├── logistic_regression/
│   ├── random_forest/
│   ├── svm/
│
├── src/
│   ├── data/
│   │   ├── VeriInceleme.py
│   │   ├── Train_test_split.py
│   │   ├── scale_features.py
│   │
│   ├── models/
│       ├── knn_model.py
│       ├── logistic_regression_model.py
│       ├── random_forest_model.py
│       ├── svm_model.py
│
└── README.md
```

---

## 📈 Results

| Model               | Accuracy |
| ------------------- | -------- |
| KNN                 | ~0.99    |
| Logistic Regression | ~0.99    |
| Random Forest       | 1.00     |
| SVM                 | ~0.99    |

---

## 🚀 Future Work

* Hyperparameter tuning for all models
* Deep learning approaches
* Real-time driver monitoring system integration

---

## 👩‍💻 Author

Developed as part of a Machine Learning course project.
