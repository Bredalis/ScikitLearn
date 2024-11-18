
import pandas as pd

# Lectura de datos
df = pd.read_csv("../CSV/HRDataset_V14.csv")
print(f"DataFrame completo:\n{df}")
print(f"\nColumnas del DataFrame:\n{df.columns}")

# Crear nuevo DataFrame con columnas seleccionadas
df = df[
  [
    "Employee_Name", "Position", 
    "State", "Sex", "DOB", "MaritalDesc", 
    "CitizenDesc", "Department", "PerformanceScore",
    "EngagementSurvey", "EmpSatisfaction"
  ]
]

print(f"\nDataFrame con columnas seleccionadas:\n{df}")
print(f"Cantidad de columnas seleccionadas: {len(df.columns)}")

# One Hot Encoding para la columna 'Sex'
sexo = pd.get_dummies(df["Sex"], prefix = "Sex")
df = df.drop(["Sex"], axis = 1)
df_2 = pd.concat([df, sexo], axis = 1)

print(f"\nColumna 'Sex' codificada en formato binario:\n{sexo}")
print(f"\nDataFrame original modificado (sin la columna 'Sex'):\n{df}")
print(f"\nNuevo DataFrame con codificación binaria añadida:\n{df_2}")