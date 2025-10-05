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
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'future_precipitation'
        
    def extract_features_from_hdf5(self, file_path):
        features = {}
        
        try:
            with h5py.File(file_path, 'r') as f:
                filename = os.path.basename(file_path)
                
                # Extract date and time from filename
                date_part = filename.split('.')[4]
                date_str = date_part.split('-')[0]
                time_str = date_part.split('-')[1][1:]
                
                dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                
                # Temporal features
                features['hour'] = dt.hour
                features['minute'] = dt.minute
                features['day_of_year'] = dt.timetuple().tm_yday
                features['month'] = dt.month
                features['day'] = dt.day
                features['day_of_week'] = dt.weekday()
                
                # Find precipitation data
                precip_data = None
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
                    # Clean data
                    precip_data = precip_data.astype(np.float32)
                    precip_data[precip_data < 0] = 0
                    precip_data[precip_data > 200] = 0
                    precip_data[np.isnan(precip_data)] = 0
                    precip_data[np.isinf(precip_data)] = 0
                    
                    # Spatial precipitation statistics
                    features['precip_mean'] = float(np.mean(precip_data))
                    features['precip_max'] = float(np.max(precip_data))
                    features['precip_min'] = float(np.min(precip_data))
                    features['precip_std'] = float(np.std(precip_data))
                    features['precip_median'] = float(np.median(precip_data))
                    
                    # Percentiles
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
                    
                    # Rain intensity distribution
                    total_pixels = precip_data.size
                    no_rain = np.sum(precip_data <= 0.1)
                    light_rain = np.sum((precip_data > 0.1) & (precip_data <= 2.5))
                    moderate_rain = np.sum((precip_data > 2.5) & (precip_data <= 10))
                    heavy_rain = np.sum((precip_data > 10) & (precip_data <= 50))
                    very_heavy_rain = np.sum(precip_data > 50)
                    
                    # Coverage ratios
                    if total_pixels > 0:
                        features['no_rain_ratio'] = float(no_rain / total_pixels)
                        features['light_rain_ratio'] = float(light_rain / total_pixels)
                        features['moderate_rain_ratio'] = float(moderate_rain / total_pixels)
                        features['heavy_rain_ratio'] = float(heavy_rain / total_pixels)
                        features['very_heavy_rain_ratio'] = float(very_heavy_rain / total_pixels)
                        features['rain_coverage'] = float((total_pixels - no_rain) / total_pixels)
                        features['active_rain_ratio'] = float((moderate_rain + heavy_rain + very_heavy_rain) / total_pixels)
                    else:
                        features['no_rain_ratio'] = 0.0
                        features['light_rain_ratio'] = 0.0
                        features['moderate_rain_ratio'] = 0.0
                        features['heavy_rain_ratio'] = 0.0
                        features['very_heavy_rain_ratio'] = 0.0
                        features['rain_coverage'] = 0.0
                        features['active_rain_ratio'] = 0.0
                    
                    # Spatial variability metrics
                    if precip_data.size > 100:
                        features['spatial_variance'] = float(np.var(precip_data))
                        features['spatial_skewness'] = float(self.calculate_skewness(precip_data))
                        features['spatial_kurtosis'] = float(self.calculate_kurtosis(precip_data))
                    else:
                        features['spatial_variance'] = 0.0
                        features['spatial_skewness'] = 0.0
                        features['spatial_kurtosis'] = 0.0
                else:
                    print(f"No precipitation data found in {filename[:50]}...")
                    return None
                
                # Quality variables with defaults
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
                            features[f'{var_name}_mean'] = 0.5
                            features[f'{var_name}_max'] = 1.0
                            features[f'{var_name}_std'] = 0.3
                    else:
                        features[f'{var_name}_mean'] = 0.5
                        features[f'{var_name}_max'] = 1.0
                        features[f'{var_name}_std'] = 0.3
                
                # Geographic coordinates (Gulf of Mexico region)
                features['lat_center'] = 29.0
                features['lon_center'] = -94.0
                features['lat_range'] = 10.0
                features['lon_range'] = 15.0
                
                features['file_timestamp'] = dt.timestamp()
                features['filename'] = filename
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
        
        return features
    
    def calculate_skewness(self, data):
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0
    
    def calculate_kurtosis(self, data):
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0
    
    def load_and_prepare_data(self, max_files=None, sample_every_n=1):
        print("Loading and processing HDF5 files...")
        
        hdf5_files = [f for f in os.listdir(self.data_dir) if f.endswith('.HDF5')]
        hdf5_files.sort()
        
        if sample_every_n > 1:
            hdf5_files = hdf5_files[::sample_every_n]
            print(f"Sampling every {sample_every_n} files")
        
        if max_files:
            hdf5_files = hdf5_files[:max_files]
        
        print(f"Processing {len(hdf5_files)} files...")
        
        all_features = []
        processed_count = 0
        
        for i, filename in enumerate(hdf5_files):
            if i % 20 == 0:
                print(f"Progress: {i+1}/{len(hdf5_files)} files processed...")
            
            file_path = os.path.join(self.data_dir, filename)
            features = self.extract_features_from_hdf5(file_path)
            
            if features:
                all_features.append(features)
                processed_count += 1
        
        if not all_features:
            raise ValueError("No features could be extracted from any file")
        
        print(f"Data extraction completed:")
        print(f"Files processed: {processed_count}/{len(hdf5_files)}")
        
        df = pd.DataFrame(all_features)
        print(f"Records: {len(df)}")
        print(f"Base features: {len(df.columns)}")
        
        # Create target variable (future precipitation)
        df = df.sort_values('file_timestamp').reset_index(drop=True)
        
        # Create cyclic temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Temporal trends (rolling windows)
        for window in [3, 6, 12]:
            df[f'precip_trend_{window}'] = df['precip_mean'].rolling(window=window, min_periods=1).mean()
            df[f'precip_std_trend_{window}'] = df['precip_std'].rolling(window=window, min_periods=1).mean()
            df[f'rain_coverage_trend_{window}'] = df['rain_coverage'].rolling(window=window, min_periods=1).mean()
        
        # Temporal changes
        df['precip_change_1h'] = df['precip_mean'].diff(1).fillna(0)
        df['precip_change_3h'] = df['precip_mean'].diff(6).fillna(0)
        
        # Persistence metrics
        df['precip_persistence'] = (df['precip_mean'] > 0.1).astype(int).rolling(window=6, min_periods=1).sum()
        
        # Target variable
        df[self.target_column] = df['precip_mean'].shift(-1)
        df['target_1h'] = df['precip_mean'].shift(-2)
        df['target_3h'] = df['precip_mean'].rolling(window=6, min_periods=1).mean().shift(-6)
        
        df = df[:-6].copy()
        
        print(f"Final dataset prepared:")
        print(f"Training records: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        
        return df
    
    def prepare_features(self, df):
        # Feature categories
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
        
        all_potential_features = (temporal_features + precipitation_features + 
                                spatial_features + quality_features + 
                                geographic_features + trend_features + change_features)
        
        self.feature_columns = [col for col in all_potential_features if col in df.columns]
        
        print(f"SELECTED FEATURES: {len(self.feature_columns)}")
        print(f"Temporal: {len([f for f in self.feature_columns if f in temporal_features])}")
        print(f"Precipitation: {len([f for f in self.feature_columns if f in precipitation_features])}")
        print(f"Spatial: {len([f for f in self.feature_columns if f in spatial_features])}")
        print(f"Quality: {len([f for f in self.feature_columns if f in quality_features])}")
        print(f"Geographic: {len([f for f in self.feature_columns if f in geographic_features])}")
        print(f"Trends: {len([f for f in self.feature_columns if f in trend_features])}")
        print(f"Changes: {len([f for f in self.feature_columns if f in change_features])}")
        
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        X = X.fillna(X.median())
        y = y.fillna(0)
        
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        y_q99 = y.quantile(0.99)
        y = np.where(y > y_q99, y_q99, y)
        
        print(f"Data prepared:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Precipitation range: {y.min():.4f} - {y.max():.4f} mm/hr")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2):
        print(f"\nTRAINING RANDOM FOREST MODEL")
        print("=" * 60)
        
        # Temporal split for time series
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Data split:")
        print(f"Training: {len(X_train)} records")
        print(f"Testing: {len(X_test)} records")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training Random Forest...")
        
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\nMODEL METRICS:")
        print(f"Training:")
        print(f"   R²: {train_r2:.4f}")
        print(f"   MSE: {train_mse:.6f}")
        print(f"   MAE: {train_mae:.6f}")
        print(f"Testing:")
        print(f"   R²: {test_r2:.4f}")
        print(f"   MSE: {test_mse:.6f}")
        print(f"   MAE: {test_mae:.6f}")
        print(f"OOB Score: {self.model.oob_score_:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTOP 15 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        # Performance by precipitation category
        print(f"\nPERFORMANCE BY PRECIPITATION CATEGORY:")
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
        categories = [
            (0, 0.1, "No rain"),
            (0.1, 2.5, "Light rain"),
            (2.5, 10, "Moderate rain"),
            (10, 50, "Heavy rain")
        ]
        
        for min_val, max_val, category in categories:
            mask = (y_true >= min_val) & (y_true < max_val)
            if np.sum(mask) > 0:
                cat_r2 = r2_score(y_true[mask], y_pred[mask])
                cat_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                print(f"{category:<15}: {np.sum(mask):4d} samples, R²={cat_r2:.3f}, MAE={cat_mae:.4f}")
    
    def save_model(self, filename='rain_prediction_model.joblib'):
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_version': '2.0',
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as: {filename}")

def main():
    print("RAIN PREDICTION MODEL TRAINING")
    print("=" * 80)
    print("Version 2.0 - Enhanced with full dataset")
    
    rain_model = RainPredictionModel()
    
    try:
        print(f"\nLoading data...")
        df = rain_model.load_and_prepare_data(
            max_files=None,
            sample_every_n=1
        )
        
        X, y = rain_model.prepare_features(df)
        
        results = rain_model.train_model(X, y, test_size=0.15)
        
        rain_model.save_model('rain_prediction_model.joblib')
        
        print(f"\nTRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Records trained: {len(X)}")
        print(f"Test R²: {results['test_r2']:.4f}")
        print(f"OOB Score: {results['oob_score']:.4f}")
        print(f"Model saved: rain_prediction_model.joblib")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()