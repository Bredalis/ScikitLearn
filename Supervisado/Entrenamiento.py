
# Librerías
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Lectura de datos
dataset = load_wine()

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target
)

# Mostrar propiedades de los datos
print(f"Dataset:\n{dataset}")
print(f"\nDatos:\n{dataset.data}")
print(f"\nEtiquetas:\n{dataset.target}")

print(f"\nCaracterísticas:\n{dataset.feature_names}")
print(f"Nombres de etiquetas: {dataset.target_names}")
print("Cantidad de clases de etiquetas:", 
      len(np.unique(dataset.target_names)))

print(f"\nX_train:\n{X_train}")
print(f"\nCantidad de X_train:\n{X_train.shape}")
print(f"\nX_test:\n{X_test}")
print(f"\nCantidad de X_test:\n{X_test.shape}")

print(f"\nY_train:\n{y_train}")
print(f"\nCantidad de Y_train:\n{y_train.shape}")
print(f"\nY_test:\n{y_test}")
print(f"\nCantidad de Y_test:\n{y_test.shape}")