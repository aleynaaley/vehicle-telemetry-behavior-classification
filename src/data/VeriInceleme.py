import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n--- Dataset Yükleniyor ---")

# dataseti oku
df = pd.read_csv("../../dataset/sero_features_4.csv")

print("Dataset boyutu:", df.shape)

print("\n--- İlk 5 Satır ---")
print(df.head())

print("\n--- Sütunlar ---")
print(df.columns.tolist())

print("\n--- Genel Bilgi ---")
print(df.info())

print("\n--- Eksik Veri Kontrolü ---")
print(df.isnull().sum())

print("\n--- Target Dağılımı ---")
print(df["Target"].value_counts())

print("\n--- Target Yüzdeleri ---")
print(df["Target"].value_counts(normalize=True) * 100)

print("\n--- Feature İstatistikleri ---")
print(df.describe())

# temiz dataseti kaydet
df.to_csv("../../outputs/dataset_clean.csv", index=False)

print("\nDataset kaydedildi: outputs/dataset.csv")

# -----------------------------
# Grafik 1: Target class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Target", data=df)
plt.title("Target Class Distribution")
plt.xlabel("Driving Behavior Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../../outputs/target_class_distribution.png")
plt.show()

# -----------------------------
# Grafik 2: Korelasyon matrisi
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("../../outputs/correlation_matrix.png")
plt.show()