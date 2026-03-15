import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

print("\n--- KNN Modeli ---")

# verileri yükle
X_train = pd.read_csv("../../outputs/X_train_scaled.csv")
X_test = pd.read_csv("../../outputs/X_test_scaled.csv")
y_train = pd.read_csv("../../outputs/y_train.csv").squeeze()
y_test = pd.read_csv("../../outputs/y_test.csv").squeeze()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# model
knn = KNeighborsClassifier(n_neighbors=5)

# training
knn.fit(X_train, y_train)

# prediction
y_pred = knn.predict(X_test)

print("\n--- Model Performansı ---")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

os.makedirs("../../outputs", exist_ok=True)
plt.savefig("../../outputs/knn_confusion_matrix.png")

plt.show()