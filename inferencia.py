from pathlib import Path

import joblib
import pandas as pd


pasta = Path(__file__).resolve().parent
modelo = joblib.load(pasta / "modelo_knn_iris.joblib")
amostras = pd.read_csv(pasta / "iris_test_samples.csv")

predicoes = modelo.predict(amostras.drop(columns="classe"))
nomes_classes = {0: "setosa", 1: "versicolor", 2: "virginica"}

for numero, classe in enumerate(predicoes, start=1):
    print(f"Amostra {numero}: classe {classe} ({nomes_classes[classe]})")

acerto = (predicoes == amostras["classe"].to_numpy()).mean() * 100
print(f"Acerto total: {acerto:.2f}%")
