import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print("\n--- Random Forest Cross Validation ---")

# verileri yükle
X_train = pd.read_csv("../../../outputs/X_train_scaled.csv")
y_train = pd.read_csv("../../../outputs/y_train.csv").squeeze()

print("X_train:", X_train.shape)

# model
model = RandomForestClassifier(random_state=42)

# cross validation 
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

print("\nCV Scores:", scores)
print("Ortalama Accuracy:", scores.mean())