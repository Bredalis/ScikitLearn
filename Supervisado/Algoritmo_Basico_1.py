
# Librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Datos

data = pd.read_csv("../CSV/Country_Stats.csv", index_col = "Country")

X = np.c_[data["GDP_per_capita"]]
y = np.c_[data["Life_satisfaction"]]

print(f"data: {data}")
print(f"x: {X}")
print(f"y: {y}")

# Modelo

model = linear_model.LinearRegression().fit(X, y)

intercept, slope = model.intercept_[0], model.coef_[0][0]
money = 35000
satisfaction = model.predict([[money]])[0][0]

print(f"Parameters: {intercept, slope}")
print(f"Happy contries: {satisfaction}")

# Grafica lineal

data.plot(
	kind = "scatter", x = "GDP_per_capita", 
	y = "Life_satisfaction", figsize = (5, 3))

X = np.linspace(0, 60000, 10000)

plt.plot(X, intercept + slope * X, "b")
plt.plot([money, money], [0, satisfaction],  "r--")
plt.plot(money, satisfaction, "ro")

# Mostrar mensaje

plt.text(50000, 3.1, r"$b = 4.85$", fontsize = 14, color = "b")
plt.text(50000, 2.2, r"$w = 4.91 \times 10^{-5}", fontsize = 14, color = "b")
plt.text(25000, 5.0, r"Prediction = 5.96", fontsize = 14, color = "b")

plt.axis([0, 60000, 0, 10])
plt.ylabel("Life Satisfaction")
plt.xlabel("GPD per capita")

plt.show()