from data1.predictor import predict_demand
from data1.dashboard import build_dashboard


# ==========================================
# 1. AI 예측
# 나중에는 아래 값들을 인터넷에서 자동으로 가져올 예정
# ==========================================

predictions = predict_demand(
    prediction_date="2026-08-01",
    temp_avg_c=29.0,
    rain_mm=0.0,
    living_population_avg=19000,
    pop_index=108.0,
    is_event=0
)


# ==========================================
# 2. 사용자가 입력하는 오늘의 누적 판매량
# ==========================================

current_sales = {
    "돼지김치찌개": 25,
    "제육볶음": 30,
    "호박된장국": 15,
    "버섯소불고기": 22
}


# ==========================================
# 3. 대시보드 데이터 생성
# ==========================================

dashboard = build_dashboard(
    predictions,
    current_sales
)


# ==========================================
# 4. 결과 출력
# ==========================================

for item in dashboard:

    print()
    print("==============================")
    print(f"메뉴: {item['menu']}")
    print(f"AI 예상 주문량: {item['predicted_servings']}인분")
    print(f"현재 누적 판매량: {item['current_sales']}인분")
    print(f"달성률: {item['achievement_rate']}%")
    print(f"예상 잔여 수요: {item['remaining_expected']}인분")
    print(f"권장 조리량: {item['recommended_servings']}인분")