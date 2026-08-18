# data1/dashboard.py

MENU_DISPLAY_NAMES = {
    "김치찌개": "돼지김치찌개",
    "제육볶음": "제육볶음",
    "된장찌개": "호박된장국",
    "냉면": "버섯소불고기"
}


def build_dashboard(predictions, current_sales):
    dashboard_results = []

    for item in predictions:

        # AI 모델 내부 메뉴명
        original_menu = item["menu"]

        # 화면에 보여줄 메뉴명
        display_menu = MENU_DISPLAY_NAMES.get(
            original_menu,
            original_menu
        )

        predicted = item["predicted_servings"]
        recommended = item["recommended_servings"]

        # 사용자가 입력한 오늘 누적 판매량
        sold = current_sales.get(display_menu, 0)

        # 달성률
        if predicted > 0:
            achievement_rate = round(
                sold / predicted * 100,
                1
            )
        else:
            achievement_rate = 0.0

        # 예상 잔여 수요
        remaining_expected = max(
            predicted - sold,
            0
        )

        dashboard_results.append({
            "menu": display_menu,
            "predicted_servings": predicted,
            "recommended_servings": recommended,
            "current_sales": sold,
            "achievement_rate": achievement_rate,
            "remaining_expected": remaining_expected,
            "top_factors": item.get("top_factors", [])
        })

    return dashboard_results