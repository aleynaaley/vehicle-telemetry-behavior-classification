import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

print("\n--- SVM Modeli ---")

# verileri yükle
X_train = pd.read_csv("../../../outputs/X_train_scaled.csv")
X_test = pd.read_csv("../../../outputs/X_test_scaled.csv")
y_train = pd.read_csv("../../../outputs/y_train.csv").squeeze()
y_test = pd.read_csv("../../../outputs/y_test.csv").squeeze()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# model
model = SVC(kernel="linear")

# training
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

print("\n--- Model Performansı ---")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

report = classification_report(y_test, y_pred)
print("\n--- Classification Report ---")
print(report)

# klasör oluştur
save_path = "../../../outputs/svm"
os.makedirs(save_path, exist_ok=True)

# metrikleri kaydet
with open(f"{save_path}/svm_metrics.txt", "w") as f:
    f.write("SVM Model Sonuçları\n")
    f.write("----------------------\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("\nMetrikler kaydedildi -> outputs/svm/svm_metrics.txt")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(f"{save_path}/svm_confusion_matrix.png")

joblib.dump(model, f"{save_path}/svm_model.pkl")
print("Model kaydedildi -> svm_model.pkl")

print("Confusion matrix kaydedildi -> outputs/svm/svm_confusion_matrix.png")

plt.show()