import pandas as pd

df = pd.read_json("data/processed/dataset.json", lines=True)

print(df.columns)