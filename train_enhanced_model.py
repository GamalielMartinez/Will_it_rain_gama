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

class EnhancedRainPredictionModel:
    def __init__(self, data_dir='./data'):
        """
        Modelo mejorado de predicción de lluvia con más datos
        """
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'future_precipitation'
        
    def extract_features_from_hdf5(self, file_path):
        """
        Extrae características meteorológicas de un archivo HDF5 con mejor manejo de errores
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
                
                # Características temporales básicas
                features['hour'] = dt.hour
                features['minute'] = dt.minute
                features['day_of_year'] = dt.timetuple().tm_yday
                features['month'] = dt.month
                features['day'] = dt.day
                features['day_of_week'] = dt.weekday()  # 0=Lunes, 6=Domingo
                
                # Buscar las variables de precipitación en diferentes rutas
                precip_data = None
                
                # Intentar diferentes rutas de datos
                precip_paths = [
                    'Grid/precipitation',
                    'Grid/precipitationCal',
                    'precipitationCal',
                    'precipitation',
                    'Grid/HQprecipitation',
                    'HQprecipitation'
                ]
                
                for path in precip_paths:
                    if path in f:
                        try:
                            precip_data = np.array(f[path])
                            break
                        except:
                            continue
                
                if precip_data is not None:
                    # Limpiar datos (eliminar valores inválidos)
                    precip_data = precip_data.astype(np.float32)
                    precip_data[precip_data < 0] = 0  # Eliminar valores negativos
                    precip_data[precip_data > 200] = 0  # Eliminar valores extremos
                    precip_data[np.isnan(precip_data)] = 0  # Eliminar NaN
                    precip_data[np.isinf(precip_data)] = 0  # Eliminar infinitos
                    
                    # Estadísticas espaciales de precipitación
                    features['precip_mean'] = float(np.mean(precip_data))
                    features['precip_max'] = float(np.max(precip_data))
                    features['precip_min'] = float(np.min(precip_data))
                    features['precip_std'] = float(np.std(precip_data))
                    features['precip_median'] = float(np.median(precip_data))
                    
                    # Percentiles para capturar distribución
                    try:
                        features['precip_p75'] = float(np.percentile(precip_data, 75))
                        features['precip_p25'] = float(np.percentile(precip_data, 25))
                        features['precip_p90'] = float(np.percentile(precip_data, 90))
                        features['precip_p10'] = float(np.percentile(precip_data, 10))
                    except:
                        features['precip_p75'] = features['precip_mean']
                        features['precip_p25'] = features['precip_mean']
                        features['precip_p90'] = features['precip_max']
                        features['precip_p10'] = features['precip_min']
                    
                    # Análisis de distribución espacial
                    total_pixels = precip_data.size
                    
                    # Categorías de intensidad de lluvia (según estándares meteorológicos)
                    no_rain = np.sum(precip_data <= 0.1)
                    light_rain = np.sum((precip_data > 0.1) & (precip_data <= 2.5))
                    moderate_rain = np.sum((precip_data > 2.5) & (precip_data <= 10))
                    heavy_rain = np.sum((precip_data > 10) & (precip_data <= 50))
                    very_heavy_rain = np.sum(precip_data > 50)
                    
                    # Ratios de cobertura
                    features['no_rain_ratio'] = float(no_rain / total_pixels) if total_pixels > 0 else 0
                    features['light_rain_ratio'] = float(light_rain / total_pixels) if total_pixels > 0 else 0
                    features['moderate_rain_ratio'] = float(moderate_rain / total_pixels) if total_pixels > 0 else 0
                    features['heavy_rain_ratio'] = float(heavy_rain / total_pixels) if total_pixels > 0 else 0
                    features['very_heavy_rain_ratio'] = float(very_heavy_rain / total_pixels) if total_pixels > 0 else 0
                    
                    # Métricas adicionales
                    features['rain_coverage'] = float((total_pixels - no_rain) / total_pixels) if total_pixels > 0 else 0
                    features['active_rain_ratio'] = float(moderate_rain + heavy_rain + very_heavy_rain) / total_pixels if total_pixels > 0 else 0
                    
                    # Análisis de variabilidad espacial
                    if precip_data.size > 100:  # Solo si hay suficientes datos
                        features['spatial_variance'] = float(np.var(precip_data))
                        features['spatial_skewness'] = float(self.calculate_skewness(precip_data))
                        features['spatial_kurtosis'] = float(self.calculate_kurtosis(precip_data))
                    else:
                        features['spatial_variance'] = 0.0
                        features['spatial_skewness'] = 0.0
                        features['spatial_kurtosis'] = 0.0
                    
                else:
                    print(f"    ❌ No se encontraron datos de precipitación en {filename[:50]}...")
                    return None  # Skip this file if no precipitation data
                
                # Variables adicionales de calidad si están disponibles
                quality_vars = {
                    'Grid/probabilityLiquidPrecipitation': 'prob_liquid',
                    'probabilityLiquidPrecipitation': 'prob_liquid',
                    'Grid/precipitationQualityIndex': 'quality',
                    'precipitationQualityIndex': 'quality',
                    'Grid/randomError': 'random_error',
                    'randomError': 'random_error'
                }
                
                for path, var_name in quality_vars.items():
                    if path in f:
                        try:
                            data = np.array(f[path])
                            data = data.astype(np.float32)
                            data[np.isnan(data)] = 0
                            data[np.isinf(data)] = 0
                            features[f'{var_name}_mean'] = float(np.mean(data))
                            features[f'{var_name}_max'] = float(np.max(data))
                            features[f'{var_name}_std'] = float(np.std(data))
                        except:
                            # Valores por defecto si hay error
                            features[f'{var_name}_mean'] = 0.5
                            features[f'{var_name}_max'] = 1.0
                            features[f'{var_name}_std'] = 0.3
                    else:
                        # Valores por defecto si no existe la variable
                        features[f'{var_name}_mean'] = 0.5
                        features[f'{var_name}_max'] = 1.0
                        features[f'{var_name}_std'] = 0.3
                
                # Coordenadas geográficas (usar valores representativos de la región)
                features['lat_center'] = 29.0  # Centro del Golfo de México
                features['lon_center'] = -94.0
                features['lat_range'] = 10.0
                features['lon_range'] = 15.0
                
                # Timestamp para ordenamiento temporal
                features['file_timestamp'] = dt.timestamp()
                features['filename'] = filename
                
        except Exception as e:
            print(f"❌ Error procesando {file_path}: {str(e)}")
            return None
        
        return features
    
    def calculate_skewness(self, data):
        """Calcula la asimetría de los datos"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0
    
    def calculate_kurtosis(self, data):
        """Calcula la curtosis de los datos"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0
    
    def load_and_prepare_data(self, max_files=None, sample_every_n=1):
        """
        Carga y prepara todos los datos de los archivos HDF5
        """
        print("🔄 Cargando y procesando archivos HDF5...")
        
        # Obtener lista de archivos
        hdf5_files = [f for f in os.listdir(self.data_dir) if f.endswith('.HDF5')]
        hdf5_files.sort()  # Ordenar cronológicamente
        
        # Aplicar sampling si se especifica
        if sample_every_n > 1:
            hdf5_files = hdf5_files[::sample_every_n]
            print(f"📊 Aplicando sampling cada {sample_every_n} archivos")
        
        if max_files:
            hdf5_files = hdf5_files[:max_files]
        
        print(f"📁 Procesando {len(hdf5_files)} archivos...")
        
        # Extraer características de todos los archivos
        all_features = []
        processed_count = 0
        
        for i, filename in enumerate(hdf5_files):
            if i % 20 == 0:
                print(f"  📦 Progreso: {i+1}/{len(hdf5_files)} archivos procesados...")
            
            file_path = os.path.join(self.data_dir, filename)
            features = self.extract_features_from_hdf5(file_path)
            
            if features:
                all_features.append(features)
                processed_count += 1
        
        if not all_features:
            raise ValueError("No se pudieron extraer características de ningún archivo")
        
        print(f"✅ Datos extraídos exitosamente:")
        print(f"   📁 Archivos procesados: {processed_count}/{len(hdf5_files)}")
        
        # Convertir a DataFrame
        df = pd.DataFrame(all_features)
        print(f"   📊 Registros: {len(df)}")
        print(f"   📋 Características base: {len(df.columns)}")
        
        # Crear variable objetivo (precipitación futura)
        df = df.sort_values('file_timestamp').reset_index(drop=True)
        
        # Crear features temporales adicionales (encoding cíclico)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Variables de tendencia temporal (rolling windows)
        for window in [3, 6, 12]:
            df[f'precip_trend_{window}'] = df['precip_mean'].rolling(window=window, min_periods=1).mean()
            df[f'precip_std_trend_{window}'] = df['precip_std'].rolling(window=window, min_periods=1).mean()
            df[f'rain_coverage_trend_{window}'] = df['rain_coverage'].rolling(window=window, min_periods=1).mean()
        
        # Variables de cambio (diferencias temporales)
        df['precip_change_1h'] = df['precip_mean'].diff(1).fillna(0)
        df['precip_change_3h'] = df['precip_mean'].diff(6).fillna(0)  # 6 períodos = 3 horas (cada 30 min)
        
        # Características de persistencia (qué tan estable es la lluvia)
        df['precip_persistence'] = (df['precip_mean'] > 0.1).astype(int).rolling(window=6, min_periods=1).sum()
        
        # Variable objetivo: precipitación promedio de los próximos períodos
        # Usar múltiples horizontes de predicción
        df[self.target_column] = df['precip_mean'].shift(-1)  # 30 minutos adelante
        df['target_1h'] = df['precip_mean'].shift(-2)  # 1 hora adelante
        df['target_3h'] = df['precip_mean'].rolling(window=6, min_periods=1).mean().shift(-6)  # 3 horas adelante
        
        # Usar el target principal (30 min adelante)
        df = df[:-6].copy()  # Eliminar últimos registros sin target
        
        print(f"🎯 Dataset final preparado:")
        print(f"   📊 Registros para entrenamiento: {len(df)}")
        print(f"   📋 Características totales: {len(df.columns)}")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepara las características para el modelo con selección inteligente
        """
        # Características potenciales organizadas por categoría
        temporal_features = [
            'hour', 'minute', 'month', 'day', 'day_of_week',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
        ]
        
        precipitation_features = [
            'precip_mean', 'precip_max', 'precip_min', 'precip_std', 'precip_median',
            'precip_p75', 'precip_p25', 'precip_p90', 'precip_p10'
        ]
        
        spatial_features = [
            'no_rain_ratio', 'light_rain_ratio', 'moderate_rain_ratio', 
            'heavy_rain_ratio', 'very_heavy_rain_ratio', 'rain_coverage',
            'active_rain_ratio', 'spatial_variance', 'spatial_skewness', 'spatial_kurtosis'
        ]
        
        quality_features = [
            'prob_liquid_mean', 'prob_liquid_max', 'prob_liquid_std',
            'quality_mean', 'quality_max', 'quality_std',
            'random_error_mean', 'random_error_max', 'random_error_std'
        ]
        
        geographic_features = [
            'lat_center', 'lon_center', 'lat_range', 'lon_range'
        ]
        
        trend_features = [
            'precip_trend_3', 'precip_trend_6', 'precip_trend_12',
            'precip_std_trend_3', 'precip_std_trend_6', 'precip_std_trend_12',
            'rain_coverage_trend_3', 'rain_coverage_trend_6', 'rain_coverage_trend_12'
        ]
        
        change_features = [
            'precip_change_1h', 'precip_change_3h', 'precip_persistence'
        ]
        
        # Combinar todas las características
        all_potential_features = (temporal_features + precipitation_features + 
                                spatial_features + quality_features + 
                                geographic_features + trend_features + change_features)
        
        # Filtrar características que realmente existen
        self.feature_columns = [col for col in all_potential_features if col in df.columns]
        
        print(f"📊 CARACTERÍSTICAS SELECCIONADAS: {len(self.feature_columns)}")
        print(f"   🕒 Temporales: {len([f for f in self.feature_columns if f in temporal_features])}")
        print(f"   🌧️ Precipitación: {len([f for f in self.feature_columns if f in precipitation_features])}")
        print(f"   🗺️ Espaciales: {len([f for f in self.feature_columns if f in spatial_features])}")
        print(f"   📈 Calidad: {len([f for f in self.feature_columns if f in quality_features])}")
        print(f"   📍 Geográficas: {len([f for f in self.feature_columns if f in geographic_features])}")
        print(f"   📊 Tendencias: {len([f for f in self.feature_columns if f in trend_features])}")
        print(f"   🔄 Cambios: {len([f for f in self.feature_columns if f in change_features])}")
        
        # Preparar datos
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Manejo robusto de valores faltantes
        X = X.fillna(X.median())  # Usar mediana en lugar de media para robustez
        y = y.fillna(0)
        
        # Verificar y limpiar valores infinitos/extremos
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        # Filtrar outliers extremos en y
        y_q99 = y.quantile(0.99)
        y = np.where(y > y_q99, y_q99, y)
        
        print(f"✅ Datos preparados:")
        print(f"   📊 Forma X: {X.shape}")
        print(f"   🎯 Forma y: {y.shape}")
        print(f"   💧 Rango precipitación: {y.min():.4f} - {y.max():.4f} mm/hr")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        """
        Entrena el modelo Random Forest con hiperparámetros optimizados
        """
        print(f"\n🤖 ENTRENANDO MODELO RANDOM FOREST MEJORADO")
        print("=" * 60)
        
        # División temporal (importante para series de tiempo)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 División de datos:")
        print(f"   🏋️ Entrenamiento: {len(X_train)} registros")
        print(f"   🧪 Prueba: {len(X_test)} registros")
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo Random Forest con parámetros optimizados
        print(f"🌳 Entrenando Random Forest...")
        
        self.model = RandomForestRegressor(
            n_estimators=200,  # Más árboles para mayor estabilidad
            max_depth=20,      # Profundidad aumentada
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Selección automática de características
            random_state=42,
            n_jobs=-1,
            oob_score=True     # Out-of-bag score para validación adicional
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Métricas detalladas
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n📈 MÉTRICAS DEL MODELO:")
        print(f"   🏋️ Entrenamiento:")
        print(f"      R²: {train_r2:.4f}")
        print(f"      MSE: {train_mse:.6f}")
        print(f"      MAE: {train_mae:.6f}")
        print(f"   🧪 Prueba:")
        print(f"      R²: {test_r2:.4f}")
        print(f"      MSE: {test_mse:.6f}")
        print(f"      MAE: {test_mae:.6f}")
        print(f"   🎯 OOB Score: {self.model.oob_score_:.4f}")
        
        # Análisis de importancia de características
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 TOP 15 CARACTERÍSTICAS MÁS IMPORTANTES:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        # Evaluación de rendimiento por categorías de precipitación
        print(f"\n🌧️ ANÁLISIS POR CATEGORÍAS DE PRECIPITACIÓN:")
        self.evaluate_by_precipitation_category(y_test, y_pred_test)
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'oob_score': self.model.oob_score_,
            'feature_importance': feature_importance,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
    
    def evaluate_by_precipitation_category(self, y_true, y_pred):
        """
        Evalúa el rendimiento por categorías de precipitación
        """
        categories = [
            (0, 0.1, "Sin lluvia"),
            (0.1, 2.5, "Lluvia ligera"),
            (2.5, 10, "Lluvia moderada"),
            (10, 50, "Lluvia fuerte")
        ]
        
        for min_val, max_val, category in categories:
            mask = (y_true >= min_val) & (y_true < max_val)
            if np.sum(mask) > 0:
                cat_r2 = r2_score(y_true[mask], y_pred[mask])
                cat_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                print(f"   {category:<15}: {np.sum(mask):4d} muestras, R²={cat_r2:.3f}, MAE={cat_mae:.4f}")
    
    def predict_rain_probability(self, lat, lon, hour=None, minute=None):
        """
        Predice precipitación usando el modelo mejorado
        """
        if self.model is None:
            return {"error": "Modelo no entrenado", "category": "❌ Error"}
        
        if hour is None or minute is None:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
        
        features = {}
        
        # Características temporales
        temporal_mappings = {
            'hour': hour,
            'minute': minute,
            'month': datetime.now().month,
            'day': datetime.now().day,
            'day_of_week': datetime.now().weekday(),
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365),
            'day_cos': np.cos(2 * np.pi * datetime.now().timetuple().tm_yday / 365),
            'month_sin': np.sin(2 * np.pi * datetime.now().month / 12),
            'month_cos': np.cos(2 * np.pi * datetime.now().month / 12),
            'dow_sin': np.sin(2 * np.pi * datetime.now().weekday() / 7),
            'dow_cos': np.cos(2 * np.pi * datetime.now().weekday() / 7)
        }
        
        # Características geográficas
        geographic_mappings = {
            'lat_center': lat,
            'lon_center': lon,
            'lat_range': 10.0,
            'lon_range': 15.0
        }
        
        # Combinar características conocidas
        features.update(temporal_mappings)
        features.update(geographic_mappings)
        
        # Valores por defecto para características meteorológicas
        default_values = {
            'precip_mean': 0.3, 'precip_max': 1.0, 'precip_min': 0.0, 
            'precip_std': 0.5, 'precip_median': 0.2, 'precip_p75': 0.8, 
            'precip_p25': 0.1, 'precip_p90': 1.2, 'precip_p10': 0.05,
            'no_rain_ratio': 0.6, 'light_rain_ratio': 0.25, 
            'moderate_rain_ratio': 0.1, 'heavy_rain_ratio': 0.04, 
            'very_heavy_rain_ratio': 0.01, 'rain_coverage': 0.4,
            'active_rain_ratio': 0.15, 'spatial_variance': 0.3,
            'spatial_skewness': 1.2, 'spatial_kurtosis': 2.5,
            'prob_liquid_mean': 0.5, 'prob_liquid_max': 0.8, 'prob_liquid_std': 0.3,
            'quality_mean': 0.7, 'quality_max': 0.9, 'quality_std': 0.2,
            'random_error_mean': 0.3, 'random_error_max': 0.6, 'random_error_std': 0.2,
            'precip_trend_3': 0.4, 'precip_trend_6': 0.35, 'precip_trend_12': 0.3,
            'precip_std_trend_3': 0.5, 'precip_std_trend_6': 0.45, 'precip_std_trend_12': 0.4,
            'rain_coverage_trend_3': 0.4, 'rain_coverage_trend_6': 0.35, 'rain_coverage_trend_12': 0.3,
            'precip_change_1h': 0.0, 'precip_change_3h': 0.0, 'precip_persistence': 2.0
        }
        
        # Llenar características faltantes
        for col in self.feature_columns:
            if col not in features:
                features[col] = default_values.get(col, 0.5)
        
        # Crear DataFrame y hacer predicción
        X_pred = pd.DataFrame([features])[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Predicción
        precip_prediction = max(0, self.model.predict(X_pred_scaled)[0])
        
        # Categorización mejorada
        if precip_prediction > 10:
            category = "🌧️ Lluvia Muy Fuerte"
            probability = "Muy Alta"
        elif precip_prediction > 2.5:
            category = "🌦️ Lluvia Fuerte"
            probability = "Alta"
        elif precip_prediction > 0.5:
            category = "🌤️ Lluvia Moderada"
            probability = "Media"
        elif precip_prediction > 0.1:
            category = "⛅ Lluvia Ligera"
            probability = "Baja"
        else:
            category = "☀️ Sin Lluvia"
            probability = "Muy Baja"
        
        return {
            "precipitation_mm_hr": round(precip_prediction, 4),
            "probability": probability,
            "category": category,
            "coordinates": {"lat": lat, "lon": lon},
            "time": f"{hour:02d}:{minute:02d}"
        }
    
    def save_model(self, filename='rain_prediction_model_enhanced.joblib'):
        """
        Guarda el modelo mejorado
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_version': '2.0',
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Modelo mejorado guardado como: {filename}")

def main():
    """
    Función principal para entrenar el modelo mejorado
    """
    print("🌧️ MODELO MEJORADO DE PREDICCIÓN DE LLUVIA")
    print("=" * 80)
    print("Version 2.0 - Con todos los datos disponibles")
    
    # Crear instancia del modelo
    rain_model = EnhancedRainPredictionModel()
    
    try:
        # Cargar datos (usar todos los archivos disponibles)
        print(f"\n📂 Cargando datos...")
        df = rain_model.load_and_prepare_data(
            max_files=None,  # Usar todos los archivos
            sample_every_n=1  # No hacer sampling, usar todos
        )
        
        # Preparar características
        X, y = rain_model.prepare_features(df)
        
        # Entrenar modelo
        results = rain_model.train_model(X, y, test_size=0.15)  # Usar menos datos para test
        
        # Guardar modelo mejorado
        rain_model.save_model('rain_prediction_model_enhanced.joblib')
        
        # También guardar como modelo principal para compatibilidad
        rain_model.save_model('rain_prediction_model.joblib')
        
        # Predicciones de ejemplo
        print(f"\n🔮 PREDICCIONES DE EJEMPLO CON MODELO MEJORADO:")
        print("=" * 60)
        
        test_coordinates = [
            (29.0, -94.0, "Houston, TX"),
            (25.5, -97.5, "Golfo de México Centro"),
            (30.5, -88.0, "Mobile, AL"),
            (27.8, -95.3, "Corpus Christi, TX"),
            (25.7617, -80.1918, "Miami, FL")
        ]
        
        for lat, lon, location in test_coordinates:
            prediction = rain_model.predict_rain_probability(lat, lon)
            print(f"  📍 {location:<25} -> {prediction['precipitation_mm_hr']:6.3f} mm/hr")
            print(f"      {prediction['category']} (Probabilidad: {prediction['probability']})")
        
        print(f"\n🎉 ¡MODELO MEJORADO ENTRENADO EXITOSAMENTE!")
        print("=" * 60)
        print(f"   📊 Registros entrenados: {len(X)}")
        print(f"   🎯 R² en prueba: {results['test_r2']:.4f}")
        print(f"   📈 OOB Score: {results['oob_score']:.4f}")
        print(f"   📁 Modelo guardado como: rain_prediction_model_enhanced.joblib")
        print(f"   ✅ Modelo principal actualizado: rain_prediction_model.joblib")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()