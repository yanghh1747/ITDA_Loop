import pandas as pd

df = pd.read_csv("data/sales_data.csv")

print("앞부분 5행")
print(df.head())

print("\n데이터 크기")
print(df.shape)

print("\n컬럼명")
print(df.columns.tolist())

print("\n결측치 개수")
print(df.isnull().sum())
