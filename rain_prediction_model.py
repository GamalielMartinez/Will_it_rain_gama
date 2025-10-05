import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime, timedelta
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
                # Formato: 3B-HHR-E.MS.MRG.3IMERG.20251004-S010000-E012959.0060.V07B.HDF5
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
                
                # Variables principales de precipitación
                if 'Grid/precipitationCal' in f:
                    precip_data = np.array(f['Grid/precipitationCal'])
                    
                    # Estadísticas espaciales de precipitación
                    features['precip_mean'] = np.nanmean(precip_data)
                    features['precip_max'] = np.nanmax(precip_data)
                    features['precip_min'] = np.nanmin(precip_data)
                    features['precip_std'] = np.nanstd(precip_data)
                    features['precip_median'] = np.nanmedian(precip_data)
                    features['precip_p75'] = np.nanpercentile(precip_data, 75)
                    features['precip_p25'] = np.nanpercentile(precip_data, 25)
                    
                    # Porcentaje de área con lluvia
                    rain_pixels = np.sum(precip_data > 0.1)  # Umbral de lluvia: 0.1 mm/hr
                    total_pixels = precip_data.size
                    features['rain_coverage'] = rain_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # Intensidad de lluvia por categorías
                    light_rain = np.sum((precip_data > 0.1) & (precip_data <= 2.5))
                    moderate_rain = np.sum((precip_data > 2.5) & (precip_data <= 10))
                    heavy_rain = np.sum(precip_data > 10)
                    
                    features['light_rain_ratio'] = light_rain / total_pixels if total_pixels > 0 else 0
                    features['moderate_rain_ratio'] = moderate_rain / total_pixels if total_pixels > 0 else 0
                    features['heavy_rain_ratio'] = heavy_rain / total_pixels if total_pixels > 0 else 0
                
                # Variables de calidad y probabilidad
                if 'Grid/probabilityLiquidPrecipitation' in f:
                    prob_data = np.array(f['Grid/probabilityLiquidPrecipitation'])
                    features['prob_liquid_mean'] = np.nanmean(prob_data)
                    features['prob_liquid_max'] = np.nanmax(prob_data)
                
                if 'Grid/precipitationQualityIndex' in f:
                    quality_data = np.array(f['Grid/precipitationQualityIndex'])
                    features['quality_mean'] = np.nanmean(quality_data)
                    features['quality_min'] = np.nanmin(quality_data)
                
                # Coordenadas (promedio del área)
                if 'Grid/lat' in f and 'Grid/lon' in f:
                    lat_data = np.array(f['Grid/lat'])
                    lon_data = np.array(f['Grid/lon'])
                    features['lat_center'] = np.mean(lat_data)
                    features['lon_center'] = np.mean(lon_data)
                    features['lat_range'] = np.max(lat_data) - np.min(lat_data)
                    features['lon_range'] = np.max(lon_data) - np.min(lon_data)
                
                features['file_timestamp'] = dt.timestamp()
                features['filename'] = filename
                
        except Exception as e:
            print(f"Error procesando {file_path}: {str(e)[:50]}...")
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
            if i % 10 == 0:
                print(f"  Procesando archivo {i+1}/{len(hdf5_files)}: {filename}")
            
            file_path = os.path.join(self.data_dir, filename)
            features = self.extract_features_from_hdf5(file_path)
            
            if features:
                all_features.append(features)
        
        if not all_features:
            raise ValueError("No se pudieron extraer características de ningún archivo")
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_features)
        print(f"✅ Datos extraídos: {len(df)} registros con {len(df.columns)} características")
        
        # Crear variable objetivo (precipitación futura)
        # Usaremos la precipitación del siguiente período de tiempo como target
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
        # Seleccionar características para el modelo
        feature_columns = [
            'hour', 'minute', 'month', 'day',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'precip_mean', 'precip_max', 'precip_std', 'precip_median',
            'precip_p75', 'precip_p25', 'rain_coverage',
            'light_rain_ratio', 'moderate_rain_ratio', 'heavy_rain_ratio',
            'prob_liquid_mean', 'prob_liquid_max', 'quality_mean',
            'lat_center', 'lon_center', 'lat_range', 'lon_range',
            'precip_trend_3', 'precip_trend_6'
        ]
        
        # Filtrar columnas que existen
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        print(f"📊 Características seleccionadas: {len(self.feature_columns)}")
        for i, feature in enumerate(self.feature_columns):
            print(f"  {i+1:2d}. {feature}")
        
        # Preparar datos
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Manejar valores faltantes
        X = X.fillna(X.mean())
        y = y.fillna(0)
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        """
        Entrena el modelo Random Forest
        """
        print(f"\n🤖 ENTRENANDO MODELO RANDOM FOREST")
        print("=" * 50)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False  # No shuffle para mantener orden temporal
        )
        
        print(f"📊 Datos de entrenamiento: {len(X_train)} registros")
        print(f"📊 Datos de prueba: {len(X_test)} registros")
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo Random Forest
        print(f"🌳 Entrenando Random Forest...")
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
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
        
        print(f"\n🎯 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<20} {row['importance']:.4f}")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': feature_importance,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_test': y_pred_test
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
        features = {
            'hour': hour,
            'minute': minute,
            'month': datetime.now().month,
            'day': datetime.now().day,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365),
            'day_cos': np.cos(2 * np.pi * datetime.now().timetuple().tm_yday / 365),
            'lat_center': lat,
            'lon_center': lon,
            'lat_range': 0.1,  # Valores por defecto
            'lon_range': 0.1,
            # Valores promedio para otras características
            'precip_mean': 0.5,
            'precip_max': 1.0,
            'precip_std': 0.3,
            'precip_median': 0.2,
            'precip_p75': 0.8,
            'precip_p25': 0.1,
            'rain_coverage': 0.3,
            'light_rain_ratio': 0.2,
            'moderate_rain_ratio': 0.1,
            'heavy_rain_ratio': 0.05,
            'prob_liquid_mean': 0.5,
            'prob_liquid_max': 0.8,
            'quality_mean': 0.7,
            'precip_trend_3': 0.4,
            'precip_trend_6': 0.3
        }
        
        # Crear DataFrame con características disponibles
        available_features = {k: v for k, v in features.items() if k in self.feature_columns}
        X_pred = pd.DataFrame([available_features])
        
        # Llenar valores faltantes
        for col in self.feature_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0.5  # Valor por defecto
        
        X_pred = X_pred[self.feature_columns]
        
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
    
    def load_model(self, filename='rain_prediction_model.joblib'):
        """
        Carga un modelo previamente guardado
        """
        model_data = joblib.load(filename)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        print(f"✅ Modelo cargado desde: {filename}")

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
        df = rain_model.load_and_prepare_data(max_files=50)  # Limitar archivos para velocidad
        
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
            rain_probability = "Alta" if prediction > 1.0 else "Media" if prediction > 0.1 else "Baja"
            print(f"  📍 {location:<25} -> {prediction:.3f} mm/hr (Probabilidad: {rain_probability})")
        
        print(f"\n✅ ¡Modelo entrenado exitosamente!")
        print(f"🎯 R² en prueba: {results['test_r2']:.4f}")
        print(f"📁 Modelo guardado como: rain_prediction_model.joblib")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()