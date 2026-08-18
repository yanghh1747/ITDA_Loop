from data1.predictor import predict_demand


results = predict_demand(
    prediction_date="2026-08-01",
    temp_avg_c=29.0,
    rain_mm=0.0,
    living_population_avg=19000,
    pop_index=108.0,
    is_event=0
)


for item in results:

    print(item)