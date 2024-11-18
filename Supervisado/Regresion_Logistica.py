
# Librerías
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Lectura de datos
dataset = load_iris()

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
	dataset.data, dataset.target, test_size = 0.25, random_state = 42
)

# Modelo
clf = LogisticRegression().fit(X_train, y_train)

# Predicción
y_pred = clf.predict(X_test)

# Resultados
print(f"Predicciones:\n{y_pred}")
print(f"Tamaño de y_pred: {y_pred.shape}")
print(f"Tamaño de y_test: {y_test.shape}")
print(f"\nMatriz de confusión:\n{confusion_matrix(y_pred, y_test)}")