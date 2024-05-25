
import pandas as pd

# Lectura de datos

df = pd.read_csv("HRDataset_v14.csv")
print(f"DF: \n{df}")
print(f"\nColumnas: \n{df.columns}")

# Crear nuevo df

df = df[[
  "Employee_Name", "Position", 
  "State", "Sex", "DOB", "MaritalDesc", 
  "CitizenDesc", "Department", "PerformanceScore",
  "EngagementSurvey", "EmpSatisfaction"
]]

print(f"DF: \n{df}")
print(f"Cantidad de columnas: {len(df.columns)}")

# One Hot Encoding

sexo = pd.get_dummies(df["Sex"], prefix = "Sex")
df = df.drop(["Sex"], axis = 1)
df_2 = pd.concat([df, sexo], axis = 1)

print(f"\nColumna con datos binarios: \n{sexo}")
print(f"\nDF (Modificada): \n{df}")
print(f"\nDF 2: \n{df_2}")