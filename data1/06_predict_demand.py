import math
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
# 3. 수요예측 함수
# =========================

def predict_demand(
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
            history["menu"] == menu
        ].copy()

        menu_history = menu_history[
            menu_history["date"] < prediction_date
        ].sort_values("date")


        # -------------------------
        # 과거 판매 데이터 확인
        # -------------------------

        if len(menu_history) < 7:
            raise ValueError(
                f"{menu}: 예측에 필요한 과거 판매 데이터가 부족합니다."
            )


        # 전날 판매량
        sales_lag_1 = (
            menu_history.iloc[-1]["qty_sold"]
        )


        # 7일 전 판매량
        sales_lag_7 = (
            menu_history.iloc[-7]["qty_sold"]
        )


        # 최근 7일 평균
        sales_rolling_7 = (
            menu_history.tail(7)["qty_sold"].mean()
        )


        # 요일
        day_of_week = prediction_date.day_name()

        # 주말 여부
        is_weekend = int(
            prediction_date.dayofweek >= 5
        )


        row = {
            "day_of_week": day_of_week,
            "menu": menu,
            "is_weekend": is_weekend,

            "temp_avg_c": temp_avg_c,
            "rain_mm": rain_mm,

            "living_population_avg":
                living_population_avg,

            "pop_index": pop_index,

            "is_event": is_event,

            "sales_lag_1": sales_lag_1,
            "sales_lag_7": sales_lag_7,
            "sales_rolling_7": sales_rolling_7,

            "month": prediction_date.month
        }

        rows.append(row)


    # =========================
    # 4. DataFrame 변환
    # =========================

    X = pd.DataFrame(rows)


    # 문자 변수 One-Hot Encoding
    X_encoded = pd.get_dummies(
        X,
        columns=[
            "day_of_week",
            "menu"
        ]
    )


    # 학습할 때 사용한 컬럼과
    # 완전히 동일하게 맞추기
    X_encoded = X_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # =========================
    # 5. AI 예측
    # =========================

    predictions = model.predict(
        X_encoded
    )


    # =========================
    # 6. 결과 생성
    # =========================

    results = []

    for menu, prediction in zip(
        X["menu"],
        predictions
    ):

        predicted_servings = max(
            0,
            round(prediction)
        )

        # 안전 여유율 5%
        recommended_servings = math.ceil(
            predicted_servings * 1.05
        )

        results.append({
            "menu": menu,
            "predicted_servings":
                predicted_servings,
            "recommended_servings":
                recommended_servings
        })


    return results


# =========================
# 7. 테스트
# =========================

if __name__ == "__main__":

    result = predict_demand(

        prediction_date="2026-08-01",

        # 아래 값은 우선 테스트용
        temp_avg_c=29.0,
        rain_mm=0.0,
        living_population_avg=19000,
        pop_index=108.0,
        is_event=0
    )

    print("======================")
    print("2026-08-01 메뉴 수요예측")
    print("======================")

    for item in result:

        print(
            f"{item['menu']} : "
            f"예상 {item['predicted_servings']}인분"
            f" → 권장 {item['recommended_servings']}인분"
        )