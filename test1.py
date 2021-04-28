import json
import pandas as pd
import requests


filename = "USD_JPY.xlsx"
df = pd.read_excel(filename)
text=df.to_json(orient='records')
resp = requests.post("http://localhost:5000/predict",json = text)
print(resp.status_code)
print(resp.json())