import pandas as pd
import os

excel_path = "data/online_Retail.xlsx"
csv_path = os.path.join("data", "online_retail.csv")

df = pd.read_excel(excel_path)

df.to_csv(csv_path, index=False)