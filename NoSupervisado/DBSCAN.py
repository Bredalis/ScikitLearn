
# Librerias

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Datos

df = pd.read_csv("Casas.csv")
print(f"df: \n {df}")

# Modelo

modelo = DBSCAN(eps = 2, min_samples = 10)

# Entrenamiento

modelo.fit_predict(df)

# Grafica

plt.figure(figsize = (7.5, 7.5))

plt.scatter(df["A"], df["B"], c = df["A"])

plt.ylabel("House Price in Pesos (1:100,000)")
plt.xlabel("Years of Building Age")
plt.box(False)
plt.show()