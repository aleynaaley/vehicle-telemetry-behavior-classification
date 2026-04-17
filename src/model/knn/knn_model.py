import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)

print("\n--- KNN Modeli ---")

# verileri yükle
X_train = pd.read_csv("../../../outputs/X_train_scaled.csv")
X_test = pd.read_csv("../../../outputs/X_test_scaled.csv")
y_train = pd.read_csv("../../../outputs/y_train.csv").squeeze()
y_test = pd.read_csv("../../../outputs/y_test.csv").squeeze()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# kayıt klasörü
save_path = "../../../outputs/knn"
os.makedirs(save_path, exist_ok=True)

# model
knn = KNeighborsClassifier(n_neighbors=5)

# training
# KNN lazy learning olduğu için eğitim aşamasında örnekleri saklar,
# asıl hesaplama tahmin sırasında yapılır.
knn.fit(X_train, y_train)

# prediction
y_pred = knn.predict(X_test)

print("\n--- Model Performansı ---")

# temel metrikler
accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# weighted
precision_weighted = precision_score(y_test, y_pred, average="weighted")
recall_weighted = recall_score(y_test, y_pred, average="weighted")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

# macro
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

# micro
precision_micro = precision_score(y_test, y_pred, average="micro")
recall_micro = recall_score(y_test, y_pred, average="micro")
f1_micro = f1_score(y_test, y_pred, average="micro")

# ekstra metrikler
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_acc)

print("\nWeighted Precision:", precision_weighted)
print("Weighted Recall:", recall_weighted)
print("Weighted F1:", f1_weighted)

print("\nMacro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1:", f1_macro)

print("\nMicro Precision:", precision_micro)
print("Micro Recall:", recall_micro)
print("Micro F1:", f1_micro)

print("\nCohen Kappa:", kappa)
print("MCC:", mcc)

report = classification_report(y_test, y_pred)
print("\n--- Classification Report ---")
print(report)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)

# class bazlı doğruluk
class_accuracy = cm.diagonal() / cm.sum(axis=1)

print("\n--- Sınıf Bazlı Doğruluklar ---")
for i, acc in enumerate(class_accuracy, start=1):
    print(f"Class {i} Accuracy: {acc:.4f}")

# metrikleri dosyaya kaydet
with open(f"{save_path}/knn_metrics.txt", "w", encoding="utf-8") as f:
    f.write("KNN Model Sonuçları\n")
    f.write("=" * 40 + "\n\n")

    f.write("Genel Metrikler\n")
    f.write("-" * 20 + "\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Balanced Accuracy: {balanced_acc}\n")
    f.write(f"Cohen Kappa: {kappa}\n")
    f.write(f"MCC: {mcc}\n\n")

    f.write("Weighted Ortalama Metrikler\n")
    f.write("-" * 30 + "\n")
    f.write(f"Precision (weighted): {precision_weighted}\n")
    f.write(f"Recall (weighted): {recall_weighted}\n")
    f.write(f"F1 Score (weighted): {f1_weighted}\n\n")

    f.write("Macro Ortalama Metrikler\n")
    f.write("-" * 27 + "\n")
    f.write(f"Precision (macro): {precision_macro}\n")
    f.write(f"Recall (macro): {recall_macro}\n")
    f.write(f"F1 Score (macro): {f1_macro}\n\n")

    f.write("Micro Ortalama Metrikler\n")
    f.write("-" * 27 + "\n")
    f.write(f"Precision (micro): {precision_micro}\n")
    f.write(f"Recall (micro): {recall_micro}\n")
    f.write(f"F1 Score (micro): {f1_micro}\n\n")

    f.write("Classification Report\n")
    f.write("-" * 25 + "\n")
    f.write(report + "\n")

    f.write("Confusion Matrix\n")
    f.write("-" * 20 + "\n")
    f.write(str(cm) + "\n\n")

    f.write("Sınıf Bazlı Doğruluklar\n")
    f.write("-" * 25 + "\n")
    for i, acc in enumerate(class_accuracy, start=1):
        f.write(f"Class {i} Accuracy: {acc:.6f}\n")

print("\nMetrikler kaydedildi -> outputs/knn/knn_metrics.txt")

# confusion matrix görseli
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{save_path}/knn_confusion_matrix.png")
print("Confusion matrix kaydedildi -> outputs/knn/knn_confusion_matrix.png")

# modeli kaydet
joblib.dump(knn, f"{save_path}/knn_model.pkl")
print("Model kaydedildi -> outputs/knn/knn_model.pkl")

plt.show()