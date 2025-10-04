import earthaccess
import xarray as xr
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Paso 1: Autenticación
auth = earthaccess.login(strategy='environment')


# Paso 2: Buscar datos meteorológicos
datasets = earthaccess.search_data(
    keywords=["precipitation", "temperature", "snow", "wind"],
    cloud_hosted=True,
    data_type=["Granule"],
    bounding_box=(-120.0, 30.0, -70.0, 50.0),  # Norteamérica
    temporal=("2023-01-01", "2023-12-31")
)

# Paso 3: Descargar archivos
downloaded_files = earthaccess.download(datasets[:3])

# Paso 4: Procesar archivos NetCDF/HDF5
features = []
targets = []

for file in downloaded_files:
    ds = xr.open_dataset(file)
    if all(var in ds.variables for var in ["temperature", "precipitation", "snow", "wind"]):
        temp = ds["temperature"].values.flatten()
        precip = ds["precipitation"].values.flatten()
        snow = ds["snow"].values.flatten()
        wind = ds["wind"].values.flatten()

        # Filtrar NaNs
        mask = ~np.isnan(temp) & ~np.isnan(precip) & ~np.isnan(snow) & ~np.isnan(wind)
        temp, precip, snow, wind = temp[mask], precip[mask], snow[mask], wind[mask]

        for t, s, w, p in zip(temp, snow, wind, precip):
            features.append([t, s, w])
            targets.append(p)

# Paso 5: Entrenar modelo
X = np.array(features)
y = np.array(targets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Paso 6: Evaluar
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE del modelo: {rmse:.2f}")
