
# Librerias

from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
  precision_score, f1_score, classification_report)

# Lectura de datos

dataset = load_wine()
print(f"Dataset:\n {dataset}")
print(f"Cantidad: {dataset.data.shape}")

# Division de datos

X_train, X_test, y_train, y_test = train_test_split(
  dataset.data, dataset.target)

print(f"X train: \n{X_train}")
print(f"X train cantidad: \n{X_train.shape}")
print(f"X test: \n{X_test}")
print(f"X test cantidad: \n{X_test.shape}")

print(f"Y train: \n{y_train}")
print(f"Y train cantidad: \n{y_train.shape}")
print(f"Y test: \n{y_test}")
print(f"Y test cantidad: \n{y_test.shape}")

# Modelo

clf = MLPClassifier()

# Entrenamiento y prediccion

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Uso de metricas

print(f"\nPrediccion: \n{y_pred}")
print(f"\nMatriz de confucion: {confusion_matrix(y_pred, y_test)}")
print(f"\nExactitud: {accuracy_score(y_pred, y_test)}")
print(f"Rendimiento: {clf.score(X_test, y_test)}")
print("\nPrecision: \n", precision_score(y_test, y_pred, average = "macro"))
print(f1_score(y_test, y_pred, average = "macro"))
print(f"\nInforme: \n{classification_report(y_pred, y_test)}")