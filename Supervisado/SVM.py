
# Librerias

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Lectura de datos

df = pd.read_csv("Wine.csv")
df = df.astype(float).fillna(0.0)

print(f"DF: \n{df}")
print(type(df))

# Division de datos

X = df["total sulfur dioxide"]
y = df.quality

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.2, random_state = 42)

# Mostrar datos

print(f"x train: \n {X_train}")
print(f"\nx test: \n {X_test}")
print(f"\ny train: \n {y_train}")
print(f"\ny test: \n {y_test}")

# Mostar cantidad de datos

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

# Escalador 

escalador = StandardScaler()

X_train_array = escalador.fit_transform(X_train.values.reshape(-1, 1))
X_train = pd.DataFrame(X_train_array, index = X_train.index)

X_test_array = escalador.transform(X_test.values.reshape(-1, 1))
X_test = pd.DataFrame(X_test_array, index = X_test.index)

# Cambiar el nombre de la columna

X_train.columns = ["StandardScaler"]
X_test.columns = ["StandardScaler"]

# Mostrar df

print(f"\nDF x train: \n{X_train}")
print(f"\nDF x test: \n{X_test}")

# Modelo

clf = SVC(kernel = "poly")

# Entrenamiento y prediccion

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"\nPrediccion: \n{y_pred}")
print(f"\nRedimiento: {clf.score(X_test, y_test)}")