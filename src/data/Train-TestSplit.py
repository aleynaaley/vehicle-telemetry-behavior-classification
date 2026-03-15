import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- Dataset Yükleniyor ---")

df = pd.read_csv("../../dataset/dataset.csv")

print("Dataset boyutu:", df.shape)

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

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

print("\nTrain class dağılımı:")
print(y_train.value_counts())

print("\nTest class dağılımı:")
print(y_test.value_counts())

# ----------------------------
# Grafik 1: Train class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title("Train Set Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.show()

# Grafik 2: Test class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y_test)
plt.title("Test Set Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.show()