import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

print("\n--- Feature Importance Analizi ---")

# modeli yükle
model = joblib.load("../../../outputs/random_forest/random_forest_model.pkl")

# train verisini yükle
X_train = pd.read_csv("../../../outputs/X_train_scaled.csv")

# klasör
save_path = "../../../outputs/random_forest"
os.makedirs(save_path, exist_ok=True)

# importance hesapla
importances = model.feature_importances_
feature_names = X_train.columns

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# top 10
top_features = feat_df.head(10)

print("\n--- En Önemli 10 Feature ---")
print(top_features)

# CSV kaydet
top_features.to_csv(f"{save_path}/top_10_features.csv", index=False)

# grafik
plt.figure(figsize=(8,5))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.gca().invert_yaxis()

plt.title("Top 10 Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig(f"{save_path}/feature_importance.png")

print("\nKaydedildi -> outputs/random_forest/feature_importance.png")

plt.show()