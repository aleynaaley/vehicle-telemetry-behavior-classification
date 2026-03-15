import pandas as pd
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

##----------------------------------------------------------
## Veri dosyalarını oku ve birleştir

folder_path = "dataset"   # kendi klasör adını yaz
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

print("Bulunan dosya sayısı:", len(csv_files))
print(csv_files)

df_list = []

for file in csv_files:
    temp_df = pd.read_csv(file)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
df.to_csv("dataset.csv", index=False)

print("Birleşik veri boyutu:", df.shape)
df.head()

## Veri hakkında genel bilgiler ve eksik değerlerin kontrolü
print(df.columns.tolist())
print(df.info())
print(df.isnull().sum())

# Hedef değişkeninin dağılımını incelelim cunku dengesiz bir veri seti olabilir
print(df["Target"].value_counts())
print(df["Target"].value_counts(normalize=True) * 100)


# feature istatistikleri
print("\n--- Feature İstatistikleri ---")
print(df.describe())

# class dağılım grafiği
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(x="Target", data=df)

plt.title("Target Class Distribution")
plt.xlabel("Driving Behavior Class")
plt.ylabel("Count")

plt.show()