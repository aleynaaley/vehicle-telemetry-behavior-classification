import os
import re
import pandas as pd

print("\n--- Model Sonuçları Toplanıyor ---")

base_path = "../../outputs"

models = [
    ("knn", "knn_metrics.txt"),
    ("logistic_regression", "logistic_regression_metrics.txt"),
    ("svm", "svm_metrics.txt"),
    ("random_forest", "random_forest_metrics.txt"),
    ("decision_tree", "decision_tree_metrics.txt"),
    ("naive_bayes", "naive_bayes_metrics.txt")
]

results = []

def extract_metric(text, name):
    match = re.search(rf"{name}: ([0-9.]+)", text)
    return float(match.group(1)) if match else None

for model_name, file_name in models:
    path = os.path.join(base_path, model_name, file_name)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    accuracy = extract_metric(content, "Accuracy")
    precision = extract_metric(content, "Precision \\(weighted\\)")
    recall = extract_metric(content, "Recall \\(weighted\\)")
    specificity = extract_metric(content, "Average Specificity")
    f1 = extract_metric(content, "F1 Score \\(weighted\\)")
    kappa = extract_metric(content, "Cohen Kappa")
    mcc = extract_metric(content, "MCC")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "Kappa": kappa,
        "MCC": mcc
    })

df = pd.DataFrame(results)
df = df.sort_values(by="Accuracy", ascending=False)

print("\n--- Model Karşılaştırma Tablosu ---")
print(df)

df.to_csv("../../outputs/model_results.csv", index=False)

print("\nKaydedildi -> outputs/model_results.csv")