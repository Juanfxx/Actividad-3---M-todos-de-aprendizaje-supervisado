# Actividad-3---M-todos-de-aprendizaje-supervisado

Codigo utilizado en el ejercicio 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Crear un conjunto de datos sintético
data = {
    'inicio': ['A', 'A', 'B', 'C', 'B', 'C'],
    'fin': ['B', 'C', 'C', 'D', 'D', 'D'],
    'tiempo': [5, 10, 2, 1, 7, 8]  # Estos son los tiempos de viaje
}

df = pd.DataFrame(data)

# Convertir las etiquetas de inicio y fin a variables categóricas
df['inicio'] = df['inicio'].astype('category').cat.codes
df['fin'] = df['fin'].astype('category').cat.codes

# Separar las características y la variable objetivo
X = df[['inicio', 'fin']]
y = df['tiempo']
# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Hacer predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio: {mse}')

# Predecir el tiempo de viaje de A a D
inicio_codificado = 0  # Código para 'A'
fin_codificado = 1     # Código para 'D'
prediccion_tiempo = modelo.predict([[inicio_codificado, fin_codificado]])
print(f'Tiempo de viaje de A a D: {prediccion_tiempo[0]} minutos')


Actividad realizada por Juan Esteban Rodriguez Culma y Hernan Alexander Betancoruth Torres 
