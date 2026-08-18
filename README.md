# ITDA LOOP

AI 기반 메뉴 수요예측 모듈

## 수요예측 모델

- Model: LightGBM
- 학습 데이터: 652건
- Validation MAE: 4.29인분
- Baseline MAE: 5.21인분

## 주요 파일

- `data1/predictor.py`
  - 수요예측 함수

- `models/demand_model.joblib`
  - 학습된 LightGBM 모델

- `data/sales_data.csv`
  - 메뉴별 과거 판매 데이터

## 사용 방법

```python
from data1.predictor import predict_demand

results = predict_demand(
    prediction_date="2026-08-01",
    temp_avg_c=29.0,
    rain_mm=0.0,
    living_population_avg=19000,
    pop_index=108.0,
    is_event=0
)

print(results)