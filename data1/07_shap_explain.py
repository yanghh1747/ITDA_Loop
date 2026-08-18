import joblib
import pandas as pd
import shap


# =========================
# 1. 최종 모델 불러오기
# =========================

model_package = joblib.load(
    "models/demand_model.joblib"
)

model = model_package["model"]
feature_columns = model_package["feature_columns"]


# =========================
# 2. 과거 판매 데이터 불러오기
# =========================

history = pd.read_csv(
    "data/sales_data.csv"
)

history = history.rename(
    columns={
        "qty_sold(판매량(인분))": "qty_sold"
    }
)

history["date"] = pd.to_datetime(
    history["date"]
)

history = history.sort_values(
    ["menu", "date"]
)


# =========================
# 3. 예측용 데이터 만들기
# =========================

def make_prediction_data(
    prediction_date,
    temp_avg_c,
    rain_mm,
    living_population_avg,
    pop_index,
    is_event
):

    prediction_date = pd.to_datetime(
        prediction_date
    )

    menus = history["menu"].unique()

    rows = []

    for menu in menus:

        menu_history = history[
            (history["menu"] == menu)
            & (history["date"] < prediction_date)
        ].sort_values("date")

        if len(menu_history) < 7:
            raise ValueError(
                f"{menu}: 과거 데이터가 부족합니다."
            )

        # 과거 판매량
        sales_lag_1 = (
            menu_history.iloc[-1]["qty_sold"]
        )

        sales_lag_7 = (
            menu_history.iloc[-7]["qty_sold"]
        )

        sales_rolling_7 = (
            menu_history.tail(7)["qty_sold"].mean()
        )

        row = {
            "day_of_week":
                prediction_date.day_name(),

            "menu":
                menu,

            "is_weekend":
                int(prediction_date.dayofweek >= 5),

            "temp_avg_c":
                temp_avg_c,

            "rain_mm":
                rain_mm,

            "living_population_avg":
                living_population_avg,

            "pop_index":
                pop_index,

            "is_event":
                is_event,

            "sales_lag_1":
                sales_lag_1,

            "sales_lag_7":
                sales_lag_7,

            "sales_rolling_7":
                sales_rolling_7,

            "month":
                prediction_date.month
        }

        rows.append(row)

    X_original = pd.DataFrame(rows)

    X_encoded = pd.get_dummies(
        X_original,
        columns=[
            "day_of_week",
            "menu"
        ]
    )

    X_encoded = X_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return X_original, X_encoded


# =========================
# 4. 실제 테스트 조건
# =========================

X_original, X_encoded = make_prediction_data(

    prediction_date="2026-08-01",

    # 이전 테스트와 동일한 조건
    temp_avg_c=29.0,
    rain_mm=0.0,
    living_population_avg=19000,
    pop_index=108.0,
    is_event=0
)


# =========================
# 5. LightGBM 예측
# =========================

predictions = model.predict(
    X_encoded
)


# =========================
# 6. SHAP 계산
# =========================

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_encoded
)


# =========================
# 7. 보기 좋은 그룹으로 합치기
# =========================

def get_group_name(feature):

    if feature in [
        "sales_lag_1",
        "sales_lag_7",
        "sales_rolling_7"
    ]:
        return "최근 판매 추세"

    elif feature in [
        "temp_avg_c",
        "rain_mm"
    ]:
        return "날씨"

    elif feature in [
        "living_population_avg",
        "pop_index"
    ]:
        return "유동인구"

    elif feature == "is_event":
        return "행사"

    elif (
        feature == "is_weekend"
        or feature == "month"
        or feature.startswith("day_of_week_")
    ):
        return "요일/시기"

    elif feature.startswith("menu_"):
        return "메뉴 특성"

    else:
        return "기타"


# =========================
# 8. 메뉴별 설명 출력
# =========================

for i in range(len(X_encoded)):

    menu = X_original.iloc[i]["menu"]

    prediction = predictions[i]

    contributions = {}

    for feature, shap_value in zip(
        feature_columns,
        shap_values[i]
    ):

        group = get_group_name(
            feature
        )

        contributions[group] = (
            contributions.get(group, 0)
            + shap_value
        )

    # 절댓값 기준 설명 비중 계산
    total_abs = sum(
        abs(value)
        for value in contributions.values()
    )

    print("\n======================")
    print(f"{menu}")
    print("======================")

    print(
        f"AI 예상 판매량: "
        f"{prediction:.1f} 인분"
    )

    print("\n예측 영향 요인")

    sorted_items = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for group, value in sorted_items:

        if total_abs > 0:
            percent = (
                abs(value)
                / total_abs
                * 100
            )
        else:
            percent = 0

        direction = "+" if value >= 0 else ""

        print(
            f"{group:12s} "
            f"{direction}{value:.2f} 인분 "
            f"({percent:.1f}%)"
        )