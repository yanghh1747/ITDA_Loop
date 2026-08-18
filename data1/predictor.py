import math
import joblib
import pandas as pd
import shap

from pathlib import Path
# 프로젝트 최상위 폴더 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# 1. 모델 불러오기
# ==========================================

MENU_DISPLAY_NAMES = {
    "김치찌개": "돼지김치찌개",
    "제육볶음": "제육볶음",
    "된장찌개": "호박된장국",
    "냉면": "버섯소불고기"
}



model_package = joblib.load(
    BASE_DIR / "models" / "demand_model.joblib"
)

model = model_package["model"]
feature_columns = model_package["feature_columns"]


# ==========================================
# 2. 판매 이력 불러오기
# ==========================================

history = pd.read_csv(
    BASE_DIR / "data" / "sales_data.csv"
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


# ==========================================
# 3. SHAP Explainer 생성
# ==========================================

explainer = shap.TreeExplainer(model)


# ==========================================
# 4. SHAP 변수 그룹 이름
# ==========================================

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


# ==========================================
# 5. 최종 수요 예측 함수
# ==========================================

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

    # 예측일 바로 전날까지 실제 판매 데이터가 있는지 확인
    required_last_date = prediction_date - pd.Timedelta(days=1)

    actual_last_date = history["date"].max()

    if actual_last_date < required_last_date:
        raise ValueError(
            f"판매 데이터가 부족합니다. "
            f"{prediction_date.date()}을 예측하려면 "
            f"최소 {required_last_date.date()}까지의 "
            f"실제 판매 데이터가 필요합니다. "
            f"현재 데이터는 {actual_last_date.date()}까지 있습니다."
        )



    menus = history["menu"].unique()

    rows = []

    # --------------------------------------
    # 메뉴별 입력데이터 생성
    # --------------------------------------

    for menu in menus:

        menu_history = history[
            (history["menu"] == menu)
            & (history["date"] < prediction_date)
        ].sort_values("date")

        if len(menu_history) < 7:
            raise ValueError(
                f"{menu}: 과거 데이터가 부족합니다."
            )

        sales_lag_1 = (
            menu_history.iloc[-1]["qty_sold"]
        )

        sales_lag_7 = (
            menu_history.iloc[-7]["qty_sold"]
        )

        sales_rolling_7 = (
            menu_history.tail(7)["qty_sold"]
            .mean()
        )

        rows.append({

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
        })


    # ======================================
    # 6. DataFrame 변환
    # ======================================

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


    # ======================================
    # 7. LightGBM 예측
    # ======================================

    predictions = model.predict(
        X_encoded
    )


    # ======================================
    # 8. SHAP 계산
    # ======================================

    shap_values = explainer.shap_values(
        X_encoded
    )


    # ======================================
    # 9. 최종 결과 생성
    # ======================================

    results = []

    for i in range(len(X_original)):

        menu = X_original.iloc[i]["menu"]

        prediction = predictions[i]

        predicted_servings = max(
            0,
            round(prediction)
        )

        recommended_servings = math.ceil(
            predicted_servings * 1.05
        )


        # SHAP 그룹별 합산
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
                + float(shap_value)
            )


        # ======================================
        # 사용자에게 보여줄 설명 변수만 선택
        # ======================================

        display_factors = {
            group: value
            for group, value in contributions.items()
            if group not in ["메뉴 특성", "기타"]
        }


        # 영향도가 큰 순서로 정렬
        sorted_factors = sorted(
            display_factors.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )


        # 화면에는 TOP 3만 표시
        top_factors = []

        for group, value in sorted_factors[:3]:

            top_factors.append({
                "factor": group,
                "effect_servings": round(value, 2),
                "direction":
                    "increase"
                    if value >= 0
                    else "decrease"
            })


        # 최종 출력
        results.append({

            "menu":
                MENU_DISPLAY_NAMES.get(
                    menu,
                    menu
                ),

            "predicted_servings":
                predicted_servings,

            "recommended_servings":
                recommended_servings,

            "top_factors":
                top_factors
        })


    return results
