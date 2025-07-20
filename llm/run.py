import os
from pathlib import Path
from tqdm import tqdm
from predict import predict_csv
import pandas as pd

# Leer datos
sellin = pd.read_csv("../data/sell-in.txt.gz", sep="\t")
sellin["date"] = pd.to_datetime(sellin["periodo"].astype(str), format="%Y%m")
df = sellin[['product_id', 'date', 'tn']]
out = df[df['product_id'].isin(pd.read_csv("../data/product_id_apredecir.csv")['product_id'])]
for pid, grp in out.groupby('product_id'):
    path = Path("input_data")/f"{pid}.csv"
    grp[['product_id','date','tn']].to_csv(path, index=False)

def batch_predict_all(input_dir, output_csv="predicciones_llmtime.csv"):
    results = []
    input_dir = Path(input_dir)

    for path in tqdm(sorted(input_dir.glob("*.csv"))):
        try:
            product_id = int(path.stem)
            pred = predict_csv(str(path))

            if pred is not None:
                results.append({"product_id": product_id, "tn": round(pred, 5)})
            else:
                print(f"⚠️ Producto {product_id}: predicción nula o inválida")

        except Exception as e:
            print(f"❌ Error con producto {path.name}: {e}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ Predicciones guardadas en {output_csv}")

batch_predict_all("./input_data")