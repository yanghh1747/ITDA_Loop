import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "data" / "sales_data.csv"

df = pd.read_csv(file_path)

menu_mapping = {
    "김치찌개": "돼지김치찌개",
    "제육볶음": "제육볶음",
    "된장찌개": "호박된장국",
    "냉면": "버섯소불고기"
}

df["menu"] = df["menu"].replace(menu_mapping)

df.to_csv(
    file_path,
    index=False,
    encoding="utf-8-sig"
)

print("메뉴명 변경 완료")
print(df["menu"].value_counts())