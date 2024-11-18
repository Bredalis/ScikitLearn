
# Librerías
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from keras.datasets import boston_housing

# División de datos
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Mostrar datos
print(f"X_train:\n{X_train}")
print(f"\nCantidad de X_train:\n{X_train.shape}")
print(f"\nX_test:\n{X_test}")
print(f"\nCantidad de X_test:\n{X_test.shape}")
print(f"\ny_train:\n{y_train}")
print(f"\nCantidad de y_train:\n{y_train.shape}")
print(f"\ny_test:\n{y_test}")
print(f"\nCantidad de y_test:\n{y_test.shape}")

# Modelo
clf = Ridge().fit(X_train, y_train)

# Predicción
y_pred = clf.predict(X_test)
print(f"\nPredicciones:\n{y_pred}")

# Predicción en DataFrame
df_y_test = pd.DataFrame(y_test, columns = ["y_test"])
df_y_pred = pd.DataFrame(y_pred, columns = ["y_pred"])
df_resultado = pd.concat([df_y_test, df_y_pred], axis = 1)

print(f"\nResultados (y_test vs y_pred):\n{df_resultado}")

# Gráfica lineal
plt.figure(figsize = (10, 6))
plt.plot(y_pred, label = "y_pred", linestyle = "--", marker = "o")
plt.plot(y_test, label = "y_test", linestyle = "-", marker = "x")
plt.title("Modelo de Regresión (Ridge)", fontsize = 16)
plt.xlabel("Índice de casas", fontsize = 12)
plt.ylabel("Precios", fontsize = 12)
plt.legend(fontsize = 12)
plt.grid(True)
plt.show()