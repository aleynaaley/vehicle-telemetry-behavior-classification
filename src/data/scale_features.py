import pandas as pd
from sklearn.preprocessing import StandardScaler

print("\n--- Scaled Veriler Hazırlanıyor ---")

# train-test verilerini oku
X_train = pd.read_csv("../../outputs/X_train.csv")
X_test = pd.read_csv("../../outputs/X_test.csv")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# scaler tanımla
scaler = StandardScaler()

# sadece train üzerinde öğren
X_train_scaled = scaler.fit_transform(X_train)

# teste aynı dönüşümü uygula
X_test_scaled = scaler.transform(X_test)

# dataframe'e çevir
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# kaydet
X_train_scaled_df.to_csv("../../outputs/X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv("../../outputs/X_test_scaled.csv", index=False)

print("\nScaled veriler kaydedildi.")
print("X_train_scaled shape:", X_train_scaled_df.shape)
print("X_test_scaled shape:", X_test_scaled_df.shape)


#####
# X → modelin baktığı özellikler (features)
# y → modelin tahmin etmeye çalıştığı şey (target)
# Ölçekleme (Scaling) ne demek 
# Veriyi aynı ölçeğe getirir.
# AccMeanX 0.25 → 0.1
# GyroStdZ 8.2 → 0.3
# Buna StandardScaler diyoruz.
#####