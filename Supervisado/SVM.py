
# Librerías
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Lectura de datos
df = pd.read_csv("../CSV/Wine.csv")

# Convertir a float y rellenar valores faltantes con 0.0
df = df.astype(float).fillna(0.0)

print(f"DataFrame cargado:\n{df.head()}")
print(f"Tipo de datos en DataFrame:\n{df.dtypes}")

# División de datos
X = df["total sulfur dioxide"]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)

# Mostrar datos
print(f"\nX_train (primeros 5 registros):\n{X_train.head()}")
print(f"\nX_test (primeros 5 registros):\n{X_test.head()}")
print(f"\ny_train (primeros 5 registros):\n{y_train.head()}")
print(f"\ny_test (primeros 5 registros):\n{y_test.head()}")

# Mostrar cantidad de datos
print(f"\nCantidad de datos en X_train: {X_train.shape}")
print(f"Cantidad de datos en y_train: {y_train.shape}")
print(f"Cantidad de datos en X_test: {X_test.shape}")
print(f"Cantidad de datos en y_test: {y_test.shape}")

# Escalar los datos
escalador = StandardScaler()

# Escalar X_train y X_test
X_train_scaled = escalador.fit_transform(X_train.values.reshape(-1, 1))
X_test_scaled = escalador.transform(X_test.values.reshape(-1, 1))

# Crear DataFrames escalados
X_train = pd.DataFrame(X_train_scaled, index = X_train.index, columns = ["StandardScaler"])
X_test = pd.DataFrame(X_test_scaled, index = X_test.index, columns = ["StandardScaler"])

# Mostrar datos escalados
print(f"\nX_train escalado (primeros 5 registros):\n{X_train.head()}")
print(f"\nX_test escalado (primeros 5 registros):\n{X_test.head()}")

# Modelo
clf = SVC(kernel = "poly", random_state = 42)

# Entrenamiento y predicción
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Mostrar resultados
print(f"\nPredicciones:\n{y_pred}")
print(f"\nRendimiento del modelo: {clf.score(X_test, y_test):.2f}")