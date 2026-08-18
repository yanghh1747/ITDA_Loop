import joblib
import pandas as pd


# =========================
# 1. 최종 모델 불러오기
# =========================

model_package = joblib.load(
    "models/demand_model.joblib"
)

model = model_package["model"]
feature_columns = model_package["feature_columns"]


# =========================
# 2. Split 중요도
# =========================

split_importance = model.booster_.feature_importance(
    importance_type="split"
)


# =========================
# 3. Gain 중요도
# =========================

gain_importance = model.booster_.feature_importance(
    importance_type="gain"
)


# =========================
# 4. 표 만들기
# =========================

importance = pd.DataFrame({
    "feature": feature_columns,
    "split": split_importance,
    "gain": gain_importance
})


# Gain을 퍼센트로 변환
importance["gain_percent"] = (
    importance["gain"]
    / importance["gain"].sum()
    * 100
)


# Gain 기준으로 정렬
importance = importance.sort_values(
    "gain",
    ascending=False
).reset_index(drop=True)


# =========================
# 5. 결과 출력
# =========================

print("======================")
print("최종 모델 변수 중요도")
print("======================")

print(
    importance[
        [
            "feature",
            "split",
            "gain",
            "gain_percent"
        ]
    ].head(15)
)


# =========================
# 6. CSV 저장
# =========================

importance.to_csv(
    "data/final_feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:")
print("data/final_feature_importance.csv")