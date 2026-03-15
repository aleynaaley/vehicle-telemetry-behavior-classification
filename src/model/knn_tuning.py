import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os

print("\n--- KNN Tuning Başladı ---")

# verileri yükle
X_train = pd.read_csv("../../outputs/X_train_scaled.csv")
y_train = pd.read_csv("../../outputs/y_train.csv").squeeze()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

k_values = range(1, 21)
mean_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    scores = cross_val_score(
        knn,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )
    
    mean_accuracy = scores.mean()
    mean_accuracies.append(mean_accuracy)
    
    print(f"k = {k} | CV Accuracy = {mean_accuracy:.4f}")

best_k = k_values[mean_accuracies.index(max(mean_accuracies))]
best_score = max(mean_accuracies)

print("\n--- En İyi Sonuç ---")
print("Best k:", best_k)
print("Best CV Accuracy:", best_score)

# grafik
os.makedirs("../../outputs", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(k_values, mean_accuracies, marker="o")
plt.title("KNN Tuning: Accuracy vs k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validated Accuracy")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.savefig("../../outputs/knn_tuning_accuracy.png")
plt.show()