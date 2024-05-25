
# Librerias

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Lectura de datos

dataset = load_diabetes()
print(f"Dataset: \n{dataset}")

# Division de datos

X_train, X_test, y_train, y_test = train_test_split(
	dataset.data, dataset.target)

# Modelo

clf = KNeighborsClassifier()

# Entrenamiento y prediccion

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Prediccion: \n{y_pred}")

print(mean_squared_error(y_pred, y_test))

# Convertir las predicciones en un df

df_y_pred = pd.DataFrame(y_pred, columns = ["y_pred"])
df_y_test = pd.DataFrame(y_test, columns = ["y_test"])

print("\nPredicciones en df: \n", pd.concat([df_y_test, df_y_pred], axis = 1))

# Grafica

plt.plot(df_y_pred)
plt.plot(df_y_test)

plt.title("Diabetes")
plt.ylabel("Y")
plt.xlabel("X")
plt.legend(["y_pred", "y_test"])
plt.grid()

plt.show()