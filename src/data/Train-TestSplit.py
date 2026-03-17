import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n---  Dataset Yükleniyor ---")

df = pd.read_csv("../../outputs/dataset_clean.csv")

print("Dataset boyutu:", df.shape)

# feature ve target ayır
X = df.drop("Target", axis=1)
y = df["Target"]

print("\n--- Train Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\n--- Train Class Dağılımı ---")
print(y_train.value_counts())

print("\n--- Test Class Dağılımı ---")
print(y_test.value_counts())


# train-test dosyalarını kaydet
X_train.to_csv("../../outputs/X_train.csv", index=False)
X_test.to_csv("../../outputs/X_test.csv", index=False)
y_train.to_csv("../../outputs/y_train.csv", index=False)
y_test.to_csv("../../outputs/y_test.csv", index=False)

print("\nTrain-test dosyaları kaydedildi.")

# -----------------------------
# Grafik 1: Train class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train)
plt.title("Train Set Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../../outputs/train_class_distribution.png")
plt.show()

# -----------------------------
# Grafik 2: Test class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_test)
plt.title("Test Set Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../../outputs/test_class_distribution.png")
plt.show()


#### output:
#---  Dataset Yükleniyor ---
#Dataset boyutu: (1102, 61)

#--- Train Test Split ---
#X_train shape: (881, 60)
#X_test shape: (221, 60)
#y_train shape: (881,)
#y_test shape: (221,)

#--- Train Class Dağılımı ---
#Target
#3    277
#2    228
#1    199
#4    177
#Name: count, dtype: int64

#--- Test Class Dağılımı ---
#Target
#3    70
#2    57
#1    50
#4    44
#Name: count, dtype: int64

#Train-test dosyaları kaydedildi.
##