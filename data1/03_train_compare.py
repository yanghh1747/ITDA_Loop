import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


# =========================
# 1. 데이터 불러오기
# =========================
df = pd.read_csv("data/model_data.csv")

df["date"] = pd.to_datetime(df["date"])

# 날짜순 정렬
df = df.sort_values(["date", "menu"]).reset_index(drop=True)


# =========================
# 2. 학습 / 테스트 기간 나누기
# =========================

# 전체 날짜 목록
unique_dates = sorted(df["date"].unique())

# 마지막 30일을 테스트 데이터로 사용
test_dates = unique_dates[-30:]

train_df = df[~df["date"].isin(test_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()

print("전체 데이터:", df.shape)
print("학습 데이터:", train_df.shape)
print("테스트 데이터:", test_df.shape)

print("\n학습 기간")
print(train_df["date"].min(), "~", train_df["date"].max())

print("\n테스트 기간")
print(test_df["date"].min(), "~", test_df["date"].max())


# =========================
# 3. 사용할 변수 선택
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
# 4. 문자 데이터를 숫자로 변환
# =========================

X_all = pd.get_dummies(
    df[features],
    columns=["day_of_week", "menu"]
)

X_train = X_all.loc[train_df.index]
X_test = X_all.loc[test_df.index]

y_train = train_df[target]
y_test = test_df[target]


# =========================
# 5. Baseline
# 최근 7일 평균을 그대로 예측값으로 사용
# =========================

baseline_pred = test_df["sales_rolling_7"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)


# =========================
# 6. LightGBM 학습
# =========================

lightgbm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbosity=-1
)

lightgbm_model.fit(
    X_train,
    y_train
)

lightgbm_pred = lightgbm_model.predict(X_test)

lightgbm_mae = mean_absolute_error(
    y_test,
    lightgbm_pred
)


# =========================
# 7. XGBoost 학습
# =========================

xgboost_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    objective="reg:squarederror"
)

xgboost_model.fit(
    X_train,
    y_train
)

xgboost_pred = xgboost_model.predict(X_test)

xgboost_mae = mean_absolute_error(
    y_test,
    xgboost_pred
)

# ==========================================
# 8 평가 함수
# ==========================================

def evaluate_model(model_name, dataset_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3)
    }


# ==========================================
# 각 모델의 Train / Test 예측
# ==========================================

# 최근 7일 평균 Baseline
baseline_train_pred = train_df["sales_rolling_7"].values
baseline_test_pred = test_df["sales_rolling_7"].values

# LightGBM
lgb_train_pred = lightgbm_model.predict(X_train)
lgb_test_pred = lightgbm_model.predict(X_test)

# XGBoost
xgb_train_pred = xgboost_model.predict(X_train)
xgb_test_pred = xgboost_model.predict(X_test)


# ==========================================
# 성능 평가
# ==========================================

evaluation_results = []

evaluation_results.append(
    evaluate_model(
        "7-day Average",
        "Train",
        y_train,
        baseline_train_pred
    )
)

evaluation_results.append(
    evaluate_model(
        "7-day Average",
        "Test",
        y_test,
        baseline_test_pred
    )
)

evaluation_results.append(
    evaluate_model(
        "XGBoost",
        "Train",
        y_train,
        xgb_train_pred
    )
)

evaluation_results.append(
    evaluate_model(
        "XGBoost",
        "Test",
        y_test,
        xgb_test_pred
    )
)

evaluation_results.append(
    evaluate_model(
        "LightGBM",
        "Train",
        y_train,
        lgb_train_pred
    )
)

evaluation_results.append(
    evaluate_model(
        "LightGBM",
        "Test",
        y_test,
        lgb_test_pred
    )
)


# ==========================================
# 결과 출력
# ==========================================

evaluation_df = pd.DataFrame(evaluation_results)

print("\n==============================")
print("모델 성능 비교")
print("==============================")
print(evaluation_df.to_string(index=False))





# =========================
# 9. 실제값 / 예측값 확인
# =========================

result = test_df[
    ["date", "menu", "qty_sold"]
].copy()

result["최근7일평균"] = baseline_pred.values
result["LightGBM예측"] = lightgbm_pred
result["XGBoost예측"] = xgboost_pred

print("\n예측 결과 일부")
print(result.head(20))


# 결과 CSV 저장
result.to_csv(
    "data/prediction_result.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n예측 결과 저장 완료:")
print("data/prediction_result.csv")

print("\n======================")
print("메뉴별 LightGBM MAE")
print("======================")

result["LightGBM오차"] = abs(
    result["qty_sold"] - result["LightGBM예측"]
)

menu_mae = (
    result.groupby("menu")["LightGBM오차"]
    .mean()
    .sort_values()
)

print(menu_mae)

importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lightgbm_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n======================")
print("LightGBM 변수 중요도 TOP 15")
print("======================")

print(importance.head(15))

# =========================
# 11. 실제값 vs 예측값 논문 스타일 시각화
# =========================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


# Train 예측
baseline_train_pred = train_df["sales_rolling_7"].values
lightgbm_train_pred = lightgbm_model.predict(X_train)
xgboost_train_pred = xgboost_model.predict(X_train)

# Test 예측
baseline_test_pred = baseline_pred.values
lightgbm_test_pred = lightgbm_pred
xgboost_test_pred = xgboost_pred


# =========================
# 색상
# =========================

train_color = "#72C9AD"   # 민트
test_color = "#F28E6B"    # 주황


# =========================
# 그래프 1개 그리는 함수
# =========================

def draw_scatter(
    ax,
    title,
    y_train,
    train_pred,
    y_test,
    test_pred
):

    # R2 계산
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # 전체 범위 계산
    min_value = min(
        y_train.min(),
        y_test.min(),
        train_pred.min(),
        test_pred.min()
    )

    max_value = max(
        y_train.max(),
        y_test.max(),
        train_pred.max(),
        test_pred.max()
    )

    # 약간의 여백
    padding = 3
    plot_min = max(0, min_value - padding)
    plot_max = max_value + padding

    # =========================
    # Train
    # =========================

    ax.scatter(
        y_train,
        train_pred,
        s=45,
        color=train_color,
        edgecolor="dimgray",
        linewidth=0.7,
        alpha=0.75,
        label="Train data"
    )

    # =========================
    # Test
    # =========================

    ax.scatter(
        y_test,
        test_pred,
        s=45,
        color=test_color,
        edgecolor="dimgray",
        linewidth=0.7,
        alpha=0.75,
        label="Test data"
    )

    # =========================
    # y = x 기준선
    # =========================

    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="y=x"
    )

    # =========================
    # 축
    # =========================

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)

    ax.set_xlabel(
        "Actual sales (servings)",
        fontsize=10,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Predicted sales (servings)",
        fontsize=10,
        fontweight="bold"  
    )

    # 제목
    ax.set_title(
        title,
        fontsize=11,
        pad=10,
        fontweight="bold"
    )

    # =========================
    # Grid
    # =========================

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.8,
        alpha=0.5
    )

    # =========================
    # 테두리
    # =========================

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    # =========================
    # R2 표시
    # =========================

    ax.text(
        0.96,
        0.16,
        f"Train R²={train_r2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=train_color,
        fontsize=10,
        fontweight="bold"
    )

    ax.text(
        0.96,
        0.08,
        f"Test R²={test_r2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=test_color,
        fontsize=10,
        fontweight="bold"
    )

    # 범례
    ax.legend(
        loc="upper left",
        fontsize=8,
        frameon=True
    )

    # 축 숫자
    ax.tick_params(
        axis="both",
        labelsize=9
    )


# =========================
# 3개 그래프 생성
# =========================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

fig.patch.set_facecolor("white")


# (a) 7-day Average
draw_scatter(
    axes[0],
    "(a) 7-day Average",
    y_train,
    baseline_train_pred,
    y_test,
    baseline_test_pred
)


# (b) XGBoost
draw_scatter(
    axes[1],
    "(b) XGBoost",
    y_train,
    xgboost_train_pred,
    y_test,
    xgboost_test_pred
)


# (c) LightGBM
draw_scatter(
    axes[2],
    "(c) LightGBM",
    y_train,
    lightgbm_train_pred,
    y_test,
    lightgbm_test_pred
)


# =========================
# 간격 조절
# =========================

plt.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.88,
    bottom=0.15,
    wspace=0.30
)


# =========================
# 저장
# =========================

plt.savefig(
    "data/actual_vs_predicted_paper_style.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()