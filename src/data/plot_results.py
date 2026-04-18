import os
import pandas as pd
import matplotlib.pyplot as plt

print("\n--- Tüm Metrik Grafikleri Oluşturuluyor ---")

df = pd.read_csv("../../outputs/model_results.csv")

save_path = "../../outputs/metric_plots"
os.makedirs(save_path, exist_ok=True)

metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score", "MCC"]

for metric in metrics:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df["Model"], df[metric])

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.3f}",
            ha="center",
            va="bottom"
        )

    plt.title(f"{metric} Comparison of Models")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.ylim(0.85, 1.01)
    plt.tight_layout()

    filename = metric.lower().replace(" ", "_") + "_comparison.png"
    plt.savefig(f"{save_path}/{filename}")
    plt.show()

print("\nGrafikler kaydedildi -> outputs/metric_plots/")