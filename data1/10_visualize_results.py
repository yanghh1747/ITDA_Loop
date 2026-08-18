import matplotlib.pyplot as plt
import numpy as np


# =========================
# 모델 성능 결과
# =========================

models = ["7-day Average", "XGBoost", "LightGBM"]

mae = [5.210, 4.414, 4.288]
rmse = [6.915, 5.798, 5.536]
r2 = [0.321, 0.523, 0.565]


# =========================
# 1. MAE 비교
# =========================

plt.figure(figsize=(8, 5))

bars = plt.bar(models, mae)

plt.title("Model MAE Comparison")
plt.ylabel("MAE (servings)")

for bar, value in zip(bars, mae):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("data/model_mae_comparison.png", dpi=300)


# =========================
# 2. RMSE 비교
# =========================

plt.figure(figsize=(8, 5))

bars = plt.bar(models, rmse)

plt.title("Model RMSE Comparison")
plt.ylabel("RMSE (servings)")

for bar, value in zip(bars, rmse):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("data/model_rmse_comparison.png", dpi=300)


# =========================
# 3. R2 비교
# =========================

plt.figure(figsize=(8, 5))

bars = plt.bar(models, r2)

plt.title("Model R2 Comparison")
plt.ylabel("R2 Score")
plt.ylim(0, 1)

for bar, value in zip(bars, r2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("data/model_r2_comparison.png", dpi=300)
plt.show()