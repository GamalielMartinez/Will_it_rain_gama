import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RainPredictionModel:
    def __init__(self, data_dir='./data'):
        """
        Inicializa el modelo de predicción de lluvia
        """
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'future_precipitation'
        
    def extract_features_from_hdf5(self, file_path):
        """
        Extrae características meteorológicas de un archivo HDF5
        """
        features = {}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Información temporal del nombre del archivo
                filename = os.path.basename(file_path)
                
                # Extraer fecha y hora del nombre del archivo
                date_part = filename.split('.')[4]  # 20251004-S010000-E012959
                date_str = date_part.split('-')[0]  # 20251004
                time_str = date_part.split('-')[1][1:]  # 010000 (quitamos la S)
                
                # Convertir a datetime
                dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                
                features['hour'] = dt.hour
                features['minute'] = dt.minute
                features['day_of_year'] = dt.timetuple().tm_yday
                features['month'] = dt.month
                features['day'] = dt.day
                
                # Buscar las variables de precipitación en diferentes rutas
                precip_data = None
                
                # Intentar diferentes rutas de datos
                precip_paths = [
                    'Grid/precipitationCal',
                    'precipitationCal',
                    'Grid/precipitation',
                    'precipitation'
                ]
                
                for path in precip_paths:
                    if path in f:
                        try:
                            precip_data = np.array(f[path])
                            print(f"    ✅ Datos encontrados en: {path}")
                            break
                        except:
                            continue
                
                if precip_data is not None:
                    # Limpiar datos (eliminar valores inválidos)
                    precip_data = precip_data.astype(np.float32)
                    precip_data[precip_data < 0] = 0  # Eliminar valores negativos
                    precip_data[np.isnan(precip_data)] = 0  # Eliminar NaN
                    precip_data[np.isinf(precip_data)] = 0  # Eliminar infinitos
                    
                    # Estadísticas espaciales de precipitación
                    features['precip_mean'] = float(np.mean(precip_data))
                    features['precip_max'] = float(np.max(precip_data))
                    features['precip_min'] = float(np.min(precip_data))
                    features['precip_std'] = float(np.std(precip_data))
                    features['precip_median'] = float(np.median(precip_data))
                    
                    # Percentiles
                    try:
                        features['precip_p75'] = float(np.percentile(precip_data, 75))
                        features['precip_p25'] = float(np.percentile(precip_data, 25))
                    except:
                        features['precip_p75'] = features['precip_mean']
                        features['precip_p25'] = features['precip_mean']
                    
                    # Porcentaje de área con lluvia
                    rain_pixels = np.sum(precip_data > 0.1)
                    total_pixels = precip_data.size
                    features['rain_coverage'] = float(rain_pixels / total_pixels) if total_pixels > 0 else 0
                    
                    # Intensidad de lluvia por categorías
                    light_rain = np.sum((precip_data > 0.1) & (precip_data <= 2.5))
                    moderate_rain = np.sum((precip_data > 2.5) & (precip_data <= 10))
                    heavy_rain = np.sum(precip_data > 10)
                    
                    features['light_rain_ratio'] = float(light_rain / total_pixels) if total_pixels > 0 else 0
                    features['moderate_rain_ratio'] = float(moderate_rain / total_pixels) if total_pixels > 0 else 0
                    features['heavy_rain_ratio'] = float(heavy_rain / total_pixels) if total_pixels > 0 else 0
                    
                else:
                    print(f"    ❌ No se encontraron datos de precipitación en {filename}")
                    # Valores por defecto
                    for key in ['precip_mean', 'precip_max', 'precip_min', 'precip_std', 
                               'precip_median', 'precip_p75', 'precip_p25', 'rain_coverage',
                               'light_rain_ratio', 'moderate_rain_ratio', 'heavy_rain_ratio']:
                        features[key] = 0.0
                
                # Intentar obtener otras variables si están disponibles
                other_vars = {
                    'Grid/probabilityLiquidPrecipitation': 'prob_liquid',
                    'probabilityLiquidPrecipitation': 'prob_liquid',
                    'Grid/precipitationQualityIndex': 'quality',
                    'precipitationQualityIndex': 'quality'
                }
                
                for path, var_name in other_vars.items():
                    if path in f:
                        try:
                            data = np.array(f[path])
                            data = data.astype(np.float32)
                            data[np.isnan(data)] = 0
                            data[np.isinf(data)] = 0
                            features[f'{var_name}_mean'] = float(np.mean(data))
                            features[f'{var_name}_max'] = float(np.max(data))
                        except:
                            features[f'{var_name}_mean'] = 0.5
                            features[f'{var_name}_max'] = 1.0
                    else:
                        features[f'{var_name}_mean'] = 0.5
                        features[f'{var_name}_max'] = 1.0
                
                # Coordenadas (usar valores fijos para la región del Golfo de México)
                features['lat_center'] = 29.0  # Golfo de México
                features['lon_center'] = -94.0
                features['lat_range'] = 10.0
                features['lon_range'] = 15.0
                
                features['file_timestamp'] = dt.timestamp()
                features['filename'] = filename
                
        except Exception as e:
            print(f"❌ Error procesando {file_path}: {str(e)}")
            return None
        
        return features
    
    def load_and_prepare_data(self, max_files=None):
        """
        Carga y prepara todos los datos de los archivos HDF5
        """
        print("🔄 Cargando y procesando archivos HDF5...")
        
        # Obtener lista de archivos
        hdf5_files = [f for f in os.listdir(self.data_dir) if f.endswith('.HDF5')]
        hdf5_files.sort()  # Ordenar cronológicamente
        
        if max_files:
            hdf5_files = hdf5_files[:max_files]
        
        print(f"📁 Procesando {len(hdf5_files)} archivos...")
        
        # Extraer características de todos los archivos
        all_features = []
        
        for i, filename in enumerate(hdf5_files):
            print(f"  📦 Archivo {i+1}/{len(hdf5_files)}: {filename[:50]}...")
            
            file_path = os.path.join(self.data_dir, filename)
            features = self.extract_features_from_hdf5(file_path)
            
            if features:
                all_features.append(features)
            
            if i >= 30:  # Limitar para no saturar
                break
        
        if not all_features:
            raise ValueError("No se pudieron extraer características de ningún archivo")
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_features)
        print(f"✅ Datos extraídos: {len(df)} registros con {len(df.columns)} características")
        print(f"📊 Columnas disponibles: {list(df.columns)}")
        
        # Crear variable objetivo (precipitación futura)
        df = df.sort_values('file_timestamp').reset_index(drop=True)
        
        # Crear features temporales adicionales
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Variables de tendencia (promedio móvil de las últimas 3 mediciones)
        df['precip_trend_3'] = df['precip_mean'].rolling(window=3, min_periods=1).mean()
        df['precip_trend_6'] = df['precip_mean'].rolling(window=6, min_periods=1).mean()
        
        # Variable objetivo: precipitación promedio del siguiente período
        df[self.target_column] = df['precip_mean'].shift(-1)
        
        # Eliminar el último registro (no tiene target)
        df = df[:-1].copy()
        
        print(f"🎯 Dataset final: {len(df)} registros para entrenamiento")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepara las características para el modelo
        """
        # Seleccionar características disponibles
        potential_features = [
            'hour', 'minute', 'month', 'day',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'precip_mean', 'precip_max', 'precip_std', 'precip_median',
            'precip_p75', 'precip_p25', 'rain_coverage',
            'light_rain_ratio', 'moderate_rain_ratio', 'heavy_rain_ratio',
            'prob_liquid_mean', 'prob_liquid_max', 'quality_mean',
            'lat_center', 'lon_center', 'lat_range', 'lon_range',
            'precip_trend_3', 'precip_trend_6'
        ]
        
        # Filtrar características que realmente existen
        self.feature_columns = [col for col in potential_features if col in df.columns]
        
        print(f"📊 Características seleccionadas: {len(self.feature_columns)}")
        for i, feature in enumerate(self.feature_columns):
            print(f"  {i+1:2d}. {feature}")
        
        # Preparar datos
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Manejar valores faltantes
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Verificar que no hay valores infinitos
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        print(f"✅ Datos preparados: X shape = {X.shape}, y shape = {y.shape}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        """
        Entrena el modelo Random Forest
        """
        print(f"\n🤖 ENTRENANDO MODELO RANDOM FOREST")
        print("=" * 50)
        
        # Dividir datos (sin shuffle para mantener orden temporal)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Datos de entrenamiento: {len(X_train)} registros")
        print(f"📊 Datos de prueba: {len(X_test)} registros")
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo Random Forest
        print(f"🌳 Entrenando Random Forest...")
        
        self.model = RandomForestRegressor(
            n_estimators=50,  # Reducir para datos pequeños
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Métricas
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n📈 MÉTRICAS DEL MODELO:")
        print(f"  Entrenamiento - R²: {train_r2:.4f}, MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")
        print(f"  Prueba       - R²: {test_r2:.4f}, MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for i, (_, row) in enumerate(feature_importance.head(min(10, len(feature_importance))).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': feature_importance
        }
    
    def predict_rain(self, lat, lon, hour=None, minute=None):
        """
        Predice precipitación para coordenadas específicas
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta train_model() primero.")
        
        # Usar hora actual si no se especifica
        if hour is None or minute is None:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
        
        # Crear características para la predicción
        features = {}
        
        # Características básicas disponibles
        if 'hour' in self.feature_columns:
            features['hour'] = hour
        if 'minute' in self.feature_columns:
            features['minute'] = minute
        if 'month' in self.feature_columns:
            features['month'] = datetime.now().month
        if 'day' in self.feature_columns:
            features['day'] = datetime.now().day
        if 'hour_sin' in self.feature_columns:
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        if 'hour_cos' in self.feature_columns:
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        if 'day_sin' in self.feature_columns:
            features['day_sin'] = np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
        if 'day_cos' in self.feature_columns:
            features['day_cos'] = np.cos(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
        if 'lat_center' in self.feature_columns:
            features['lat_center'] = lat
        if 'lon_center' in self.feature_columns:
            features['lon_center'] = lon
        
        # Llenar el resto con valores por defecto
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.5
        
        # Crear DataFrame
        X_pred = pd.DataFrame([features])[self.feature_columns]
        
        # Escalar y predecir
        X_pred_scaled = self.scaler.transform(X_pred)
        prediction = self.model.predict(X_pred_scaled)[0]
        
        return max(0, prediction)  # No permitir precipitación negativa
    
    def save_model(self, filename='rain_prediction_model.joblib'):
        """
        Guarda el modelo entrenado
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Modelo guardado como: {filename}")

def main():
    """
    Función principal para entrenar y probar el modelo
    """
    print("🌧️ MODELO DE PREDICCIÓN DE LLUVIA CON RANDOM FOREST")
    print("=" * 80)
    
    # Crear instancia del modelo
    rain_model = RainPredictionModel()
    
    try:
        # Cargar datos
        df = rain_model.load_and_prepare_data(max_files=30)
        
        # Preparar características
        X, y = rain_model.prepare_features(df)
        
        # Entrenar modelo
        results = rain_model.train_model(X, y)
        
        # Guardar modelo
        rain_model.save_model()
        
        # Hacer algunas predicciones de ejemplo
        print(f"\n🔮 PREDICCIONES DE EJEMPLO:")
        print("=" * 40)
        
        # Coordenadas del Golfo de México
        test_coordinates = [
            (29.0, -94.0, "Houston, TX"),
            (25.5, -97.5, "Golfo de México Centro"),
            (30.5, -88.0, "Mobile, AL"),
            (27.8, -95.3, "Corpus Christi, TX")
        ]
        
        for lat, lon, location in test_coordinates:
            prediction = rain_model.predict_rain(lat, lon)
            rain_level = "Alta" if prediction > 1.0 else "Media" if prediction > 0.1 else "Baja"
            print(f"  📍 {location:<25} -> {prediction:.3f} mm/hr (Intensidad: {rain_level})")
        
        print(f"\n✅ ¡Modelo entrenado exitosamente!")
        print(f"🎯 R² en prueba: {results['test_r2']:.4f}")
        print(f"📁 Modelo guardado como: rain_prediction_model.joblib")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()