import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

print("\n--- Train ve Test Verileri Yükleniyor ---")

X_train = pd.read_csv("../../outputs/X_train.csv")
X_test = pd.read_csv("../../outputs/X_test.csv")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# scaler oluştur
scaler = StandardScaler()

print("\n--- Scaling uygulanıyor ---")

# sadece train üzerinde öğren
X_train_scaled = scaler.fit_transform(X_train)

# test setine aynı dönüşümü uygula
X_test_scaled = scaler.transform(X_test)

# tekrar dataframe yap
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# outputs klasörü
os.makedirs("../../outputs", exist_ok=True)

# kaydet
X_train_scaled.to_csv("../../outputs/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("../../outputs/X_test_scaled.csv", index=False)

print("\nScaled veriler kaydedildi.")

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)


#####
# X → modelin baktığı özellikler (features)
# y → modelin tahmin etmeye çalıştığı şey (target)
# Ölçekleme (Scaling) ne demek 
# Veriyi aynı ölçeğe getirir.
# AccMeanX 0.25 → 0.1
# GyroStdZ 8.2 → 0.3
# Buna StandardScaler diyoruz.
#####

###### output

#--- Train ve Test Verileri Yükleniyor ---
#X_train shape: (881, 60)
#X_test shape: (221, 60)

#--- Scaling uygulanıyor ---

#Scaled veriler kaydedildi.
#X_train_scaled shape: (881, 60)
#X_test_scaled shape: (221, 60)
###