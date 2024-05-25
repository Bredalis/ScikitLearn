
# Librerias

from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Lectura de datos

dataset = load_iris()

# Division de datos

X_train, X_test, y_train, y_test = train_test_split(
	dataset.data, dataset.target)

# Modelo

clf = LogisticRegression()

# Entrenamiento

clf.fit(X_train, y_train)

# Prediccion

y_pred = clf.predict(X_test)

print(f"Prediccion: \n{y_pred}")
print(y_pred.shape)
print(y_test.shape)
print(f"\nMatriz de confusion: \n{confusion_matrix(y_pred, y_test)}")