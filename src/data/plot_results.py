import pandas as pd
import matplotlib.pyplot as plt

print("\n--- Grafik Oluşturuluyor ---")

df = pd.read_csv("../../outputs/model_results.csv")

# accuracy grafiği
plt.figure(figsize=(8,5))
bars = plt.bar(df["Model"], df["Accuracy"])

# değerleri üstüne yaz
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.3f}", 
             ha='center', va='bottom')

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.xticks(rotation=45)
plt.ylim(0.90, 1.01)  # zoom effect

plt.tight_layout()
plt.savefig("../../outputs/model_accuracy_comparison.png")
plt.show()