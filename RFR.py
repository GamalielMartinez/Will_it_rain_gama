import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Cargar los datos
df = pd.read_csv('datos_clima.csv')

# 2. Definir las variables predictoras (X) y la variable objetivo (y)
X = df[['temperatura', 'precipitacion', 'humedad', 'viento']]
y = df['lluvia']  # cantidad de lluvia en mm

# 3. Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear el modelo de Random Forest Regressor
modelo = RandomForestRegressor(
    n_estimators=100,      # número de árboles
    max_depth=None,        # profundidad ilimitada
    random_state=42
)

# 5. Entrenar el modelo
modelo.fit(X_train, y_train)

# 6. Realizar predicciones
y_pred = modelo.predict(X_test)

# 7. Evaluar el modelo (métricas de regresión)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("🔍 Evaluación del modelo:")
print(f"Error absoluto medio (MAE): {mae:.3f}")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.3f}")
print(f"Coeficiente de determinación (R²): {r2:.3f}")

# 8. Comparar valores reales vs predichos
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Lluvia real (mm)")
plt.ylabel("Lluvia predicha (mm)")
plt.title("Comparación entre valores reales y predichos")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 9. Importancia de las variables
importancias = modelo.feature_importances_
features = X.columns

plt.barh(features, importancias, color='skyblue')
plt.title("Importancia de las variables")
plt.xlabel("Importancia")
plt.show()