
# Librerías

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from keras.datasets import boston_housing

# Division de datos

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Mostrar datos

print(f"X train: \n{X_train}")
print(f"\nX train cantidad: \n{X_train.shape}")
print(f"\nX test: \n{X_test}")
print(f"\nX test cantidad: \n{X_test.shape}")
print(f"\nY train: \n{y_train}")
print(f"\nY train cantidad: \n{y_train.shape}")
print(f"\nY test: \n{y_test}")
print(f"\nY test cantidad: \n{y_test.shape}")

# Modelo

clf = Ridge().fit(X_train, y_train)

# Prediccion

y_pred = clf.predict(X_test)
print(f"\nPrediccion: \n{y_pred}")

# Prediccion en df

df_y_test = pd.DataFrame(y_test, columns = ["y_test"])
df_y_pred = pd.DataFrame(y_pred, columns = ["y_pred"])

print(f"\nConcatenacion: \n{pd.concat([df_y_test, df_y_pred], axis = 1)}")

# Grafica lineal

plt.plot(y_pred)
plt.plot(y_test)

plt.title("Modelo de Regresion (Ridge)")
plt.xlabel("Casas")
plt.ylabel("Precios")
plt.legend(["y_pred", "y_test"])
plt.show()