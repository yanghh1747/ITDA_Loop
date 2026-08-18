import pandas as pd

# 1. 원본 데이터 불러오기
df = pd.read_csv("data/sales_data.csv")

# 2. 판매량 컬럼 이름을 간단하게 변경
df = df.rename(columns={
    "qty_sold(판매량(인분))": "qty_sold"
})

# 3. 날짜를 날짜 형식으로 변환
df["date"] = pd.to_datetime(df["date"])

# 4. 메뉴별, 날짜순으로 정렬
df = df.sort_values(["menu", "date"])

# 5. 과거 판매량 변수 만들기

# 전날 같은 메뉴 판매량
df["sales_lag_1"] = (
    df.groupby("menu")["qty_sold"]
      .shift(1)
)

# 7일 전 같은 메뉴 판매량
df["sales_lag_7"] = (
    df.groupby("menu")["qty_sold"]
      .shift(7)
)

# 직전 7일간 평균 판매량
df["sales_rolling_7"] = (
    df.groupby("menu")["qty_sold"]
      .transform(
          lambda x: x.shift(1).rolling(7).mean()
      )
)

# 6. 날짜에서 월 정보 추가
df["month"] = df["date"].dt.month

# 7. lag 계산 때문에 생긴 결측행 제거
df_model = df.dropna().copy()

# 8. 결과 저장
df_model.to_csv(
    "data/model_data.csv",
    index=False,
    encoding="utf-8-sig"
)

# 결과 확인
print("파생변수 생성 완료!")

print("\n데이터 크기")
print(df_model.shape)

print("\n컬럼명")
print(df_model.columns.tolist())

print("\n앞부분 10행")
print(
    df_model[
        [
            "date",
            "menu",
            "qty_sold",
            "sales_lag_1",
            "sales_lag_7",
            "sales_rolling_7"
        ]
    ].head(10)
)

print("\n결측치")
print(df_model.isnull().sum())
