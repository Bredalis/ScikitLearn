
# Librerías
from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
  precision_score, f1_score, classification_report)

# Lectura de datos
dataset = load_wine()
print(f"Dataset:\n{dataset}")
print(f"Cantidad de datos: {dataset.data.shape}")

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size = 0.25, random_state = 42
)

print(f"X_train:\n{X_train}")
print(f"Cantidad de X_train:\n{X_train.shape}")
print(f"X_test:\n{X_test}")
print(f"Cantidad de X_test:\n{X_test.shape}")

print(f"y_train:\n{y_train}")
print(f"Cantidad de y_train:\n{y_train.shape}")
print(f"y_test:\n{y_test}")
print(f"Cantidad de y_test:\n{y_test.shape}")

# Modelo
clf = MLPClassifier(random_state = 42, max_iter = 500)

# Entrenamiento y predicción
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Uso de métricas
print(f"\nPredicciones:\n{y_pred}")
print(f"\nMatriz de confusión:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nExactitud: {accuracy_score(y_test, y_pred)}")
print(f"Rendimiento del modelo: {clf.score(X_test, y_test)}")
print(f"\nPrecisión: {precision_score(y_test, y_pred, average = "macro")}")
print(f"F1 Score: {f1_score(y_test, y_pred, average = "macro")}")
print(f"\nInforme de clasificación:\n{classification_report(y_test, y_pred)}")