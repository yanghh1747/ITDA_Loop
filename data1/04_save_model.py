import os
import joblib
import pandas as pd

from lightgbm import LGBMRegressor


# =========================
# 1. 데이터 불러오기
# =========================

df = pd.read_csv("data/model_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["date", "menu"]
).reset_index(drop=True)


# =========================
# 2. 모델 입력 변수
# =========================

features = [
    "day_of_week",
    "menu",
    "is_weekend",
    "temp_avg_c",
    "rain_mm",
    "living_population_avg",
    "pop_index",
    "is_event",
    "sales_lag_1",
    "sales_lag_7",
    "sales_rolling_7",
    "month"
]

target = "qty_sold"


# =========================
# 3. 문자 데이터 숫자로 변환
# =========================

X = pd.get_dummies(
    df[features],
    columns=[
        "day_of_week",
        "menu"
    ]
)

y = df[target]


# =========================
# 4. 최종 LightGBM 모델 학습
# =========================

model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbosity=-1
)

model.fit(X, y)


# =========================
# 5. models 폴더 생성
# =========================

os.makedirs(
    "models",
    exist_ok=True
)


# =========================
# 6. 모델 + 컬럼 정보 저장
# =========================

model_package = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "features": features,
    "last_training_date": str(
        df["date"].max().date()
    )
}

joblib.dump(
    model_package,
    "models/demand_model.joblib"
)


# =========================
# 7. 확인
# =========================

print("최종 LightGBM 모델 저장 완료!")

print(
    "저장 위치:",
    "models/demand_model.joblib"
)

print(
    "학습 데이터:",
    len(df),
    "건"
)

print(
    "마지막 학습 날짜:",
    df["date"].max().date()
)

print(
    "입력 변수 개수:",
    len(X.columns)
)