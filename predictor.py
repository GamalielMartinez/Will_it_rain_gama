import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

class ConsoleRainPredictor:
    def __init__(self):
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        model_paths = [
            'rain_prediction_model.joblib',
            'src/rain_prediction_model.joblib'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    print(f"Loading model from: {path}")
                    self.model_data = joblib.load(path)
                    print(f"Model loaded successfully")
                    print(f"Version: {self.model_data.get('model_version', 'N/A')}")
                    print(f"Training date: {self.model_data.get('training_date', 'N/A')}")
                    print(f"Features: {len(self.model_data['feature_columns'])}")
                    return
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
        
        print("Could not load any model")
        sys.exit(1)
    
    def predict_rain(self, lat, lon, hour=None, minute=None):
        if self.model_data is None:
            return {
                "precipitation_mm_hr": 0.0,
                "probability": "Error - Model not loaded",
                "category": "Error",
                "coordinates": {"lat": lat, "lon": lon},
                "time": "N/A"
            }
        
        if hour is None or minute is None:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
        
        features = {}
        
        # Temporal features
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
        
        # Geographic features
        lat_factor = (lat - 25.0) / 10.0
        lon_factor = (lon + 90.0) / 10.0
        hour_factor = hour / 24.0
        
        coastal_factor = 1.0 - min(abs(lat - 29.0) / 5.0, 1.0)
        tropical_factor = max(0, 1.0 - (lat - 20.0) / 15.0)
        
        base_precip = 0.1 + (coastal_factor * 0.2) + (tropical_factor * 0.15)
        location_variance = abs(lat_factor * lon_factor) * 0.3
        temporal_variance = np.sin(2 * np.pi * hour_factor) * 0.1
        
        features['lat_center'] = lat
        features['lon_center'] = lon
        features['lat_range'] = 10.0
        features['lon_range'] = 15.0
        
        # Dynamic meteorological values
        default_values = {
            'precip_mean': base_precip + location_variance + temporal_variance,
            'precip_max': (base_precip + location_variance) * 3.0 + abs(np.sin(lat * 0.1)) * 0.5,
            'precip_min': max(0, base_precip - 0.05 + np.cos(lon * 0.1) * 0.02),
            'precip_std': 0.3 + location_variance + abs(np.sin(lat * lon * 0.01)) * 0.2,
            'precip_median': base_precip + location_variance * 0.7,
            'precip_p75': base_precip * 2.0 + location_variance + tropical_factor * 0.1,
            'precip_p25': base_precip * 0.5 + np.cos(lat * 0.1) * 0.05,
            'precip_p90': base_precip * 3.5 + coastal_factor * 0.2,
            'precip_p10': max(0, base_precip * 0.3 + np.sin(lon * 0.1) * 0.02),
            'no_rain_ratio': max(0.3, 0.8 - coastal_factor * 0.3 - tropical_factor * 0.2),
            'light_rain_ratio': 0.15 + coastal_factor * 0.1 + temporal_variance * 0.5,
            'moderate_rain_ratio': 0.08 + tropical_factor * 0.1 + coastal_factor * 0.05,
            'heavy_rain_ratio': 0.03 + tropical_factor * 0.07 + max(0, np.sin(hour_factor * np.pi) * 0.02),
            'very_heavy_rain_ratio': 0.01 + tropical_factor * 0.03,
            'rain_coverage': 0.2 + coastal_factor * 0.2 + tropical_factor * 0.15,
            'active_rain_ratio': 0.1 + tropical_factor * 0.1 + coastal_factor * 0.08,
            'spatial_variance': 0.2 + location_variance * 2.0,
            'spatial_skewness': 1.0 + np.sin(lat * lon * 0.01) * 0.5,
            'spatial_kurtosis': 2.0 + abs(lat_factor - lon_factor) * 1.0,
            'prob_liquid_mean': 0.4 + tropical_factor * 0.3 + coastal_factor * 0.1,
            'prob_liquid_max': 0.7 + tropical_factor * 0.2,
            'prob_liquid_std': 0.25 + location_variance,
            'quality_mean': 0.6 + coastal_factor * 0.2,
            'quality_max': 0.8 + coastal_factor * 0.15,
            'quality_std': 0.15 + location_variance * 0.5,
            'random_error_mean': 0.25 + location_variance * 0.2,
            'random_error_max': 0.5 + location_variance * 0.3,
            'random_error_std': 0.15 + abs(lat_factor) * 0.1,
            'precip_trend_3': base_precip * 1.2 + temporal_variance,
            'precip_trend_6': base_precip * 1.1 + temporal_variance * 0.8,
            'precip_trend_12': base_precip * 1.0 + temporal_variance * 0.6,
            'precip_std_trend_3': 0.4 + location_variance * 1.5,
            'precip_std_trend_6': 0.35 + location_variance * 1.2,
            'precip_std_trend_12': 0.3 + location_variance,
            'rain_coverage_trend_3': 0.3 + coastal_factor * 0.15 + temporal_variance,
            'rain_coverage_trend_6': 0.25 + coastal_factor * 0.12 + temporal_variance * 0.7,
            'rain_coverage_trend_12': 0.2 + coastal_factor * 0.1 + temporal_variance * 0.5,
            'precip_change_1h': np.sin(hour_factor * 2 * np.pi) * 0.05 + location_variance * 0.1,
            'precip_change_3h': np.cos(hour_factor * np.pi) * 0.03 + tropical_factor * 0.02,
            'precip_persistence': 1.0 + coastal_factor * 1.5 + tropical_factor * 1.0,
            'day_of_week': datetime.now().weekday(),
            'month_sin': np.sin(2 * np.pi * datetime.now().month / 12),
            'month_cos': np.cos(2 * np.pi * datetime.now().month / 12),
            'dow_sin': np.sin(2 * np.pi * datetime.now().weekday() / 7),
            'dow_cos': np.cos(2 * np.pi * datetime.now().weekday() / 7)
        }
        
        features.update(temporal_mappings)
        
        for col in self.model_data['feature_columns']:
            if col not in features:
                features[col] = default_values.get(col, 0.5)
        
        # Make prediction
        X_pred = pd.DataFrame([features])[self.model_data['feature_columns']]
        X_pred_scaled = self.model_data['scaler'].transform(X_pred)
        
        precip_prediction = max(0, self.model_data['model'].predict(X_pred_scaled)[0])
        
        # Calculate probability
        threshold = 0.1
        scale_factor = 10.0
        
        if precip_prediction <= 0.01:
            rain_probability = 0.0
        else:
            sigmoid_input = (precip_prediction - threshold) * scale_factor
            rain_probability = 1 / (1 + np.exp(-sigmoid_input))
            rain_probability = min(max(rain_probability, 0.0), 1.0)
        
        intensity_probability = min(precip_prediction / 5.0, 1.0)
        final_probability = (rain_probability * 0.7 + intensity_probability * 0.3)
        final_probability = min(max(final_probability, 0.0), 1.0)
        
        # Categorize prediction
        if final_probability >= 0.9:
            category = "Heavy Rain"
        elif final_probability >= 0.7:
            category = "Moderate Rain"
        elif final_probability >= 0.4:
            category = "Light Rain"
        elif final_probability >= 0.1:
            category = "Drizzle"
        else:
            category = "No Rain"
        
        probability_percent = f"{final_probability * 100:.2f}%"
        
        return {
            "precipitation_mm_hr": round(precip_prediction, 4),
            "probability": probability_percent,
            "probability_decimal": round(final_probability, 4),
            "category": category,
            "coordinates": {"lat": lat, "lon": lon},
            "time": f"{hour:02d}:{minute:02d}"
        }
    
    def print_header(self):
        print("\n" + "="*80)
        print("INTERACTIVE RAIN PREDICTOR - CONSOLE")
        print("="*80)
        print("Enter coordinates to get precipitation predictions")
        print("Use current date/time or specify custom time")
        print("Type 'exit', 'quit' or 'q' to exit")
        print("="*80)
    
    def print_prediction(self, result, location_name=None):
        print("\n" + "─"*60)
        print(f"RAIN PREDICTION")
        print("─"*60)
        
        if location_name:
            print(f"Location: {location_name}")
        
        print(f"Coordinates: {result['coordinates']['lat']:.4f}, {result['coordinates']['lon']:.4f}")
        print(f"Time: {result['time']}")
        print(f"Precipitation: {result['precipitation_mm_hr']} mm/hr")
        print(f"Probability: {result['probability']} ({result['probability_decimal']:.4f})")
        print(f"Category: {result['category']}")
        
        # Visual probability bar
        prob_decimal = result['probability_decimal']
        bar_length = 30
        filled_length = int(bar_length * prob_decimal)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"Confidence: [{bar}] {result['probability']}")
        
        print("─"*60)
    
    def get_coordinates_input(self):
        while True:
            try:
                coord_input = input("\nEnter coordinates (lat,lon) or city name: ").strip()
                
                if coord_input.lower() in ['exit', 'quit', 'q']:
                    return None, None, None
                
                # Predefined cities
                if ',' not in coord_input or not any(c.isdigit() or c == '.' or c == '-' for c in coord_input):
                    cities = {
                        'houston': (29.7604, -95.3698, "Houston, TX"),
                        'miami': (25.7617, -80.1918, "Miami, FL"),
                        'new orleans': (29.9511, -90.0715, "New Orleans, LA"),
                        'tampa': (27.9506, -82.4572, "Tampa, FL"),
                        'galveston': (29.2694, -94.7847, "Galveston, TX"),
                        'mobile': (30.6954, -88.0399, "Mobile, AL"),
                        'corpus christi': (27.8006, -97.3964, "Corpus Christi, TX"),
                        'pensacola': (30.4213, -87.2169, "Pensacola, FL"),
                        'biloxi': (30.3960, -88.8853, "Biloxi, MS"),
                        'beaumont': (30.0802, -94.1266, "Beaumont, TX")
                    }
                    
                    city_lower = coord_input.lower()
                    if city_lower in cities:
                        lat, lon, name = cities[city_lower]
                        print(f"City found: {name}")
                        return lat, lon, name
                    else:
                        print("City not found. Available cities:")
                        for city, (lat, lon, name) in cities.items():
                            print(f"   - {city} ({name})")
                        continue
                
                # Parse numeric coordinates
                parts = coord_input.split(',')
                if len(parts) != 2:
                    print("Invalid format. Use: latitude,longitude (e.g. 29.7604,-95.3698)")
                    continue
                
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                
                # Validate ranges for Gulf of Mexico region
                if not (20 <= lat <= 35):
                    print("Latitude out of valid range (20-35°N)")
                    continue
                
                if not (-100 <= lon <= -75):
                    print("Longitude out of valid range (-100 to -75°W)")
                    continue
                
                return lat, lon, None
                
            except ValueError:
                print("Invalid coordinates. Use decimal numbers (e.g. 29.7604,-95.3698)")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return None, None, None
    
    def get_time_input(self):
        try:
            time_input = input("Time (HH:MM, Enter for current): ").strip()
            
            if not time_input:
                return None, None
            
            if ':' not in time_input:
                print("Invalid time format. Use HH:MM")
                return None, None
            
            parts = time_input.split(':')
            if len(parts) != 2:
                print("Invalid time format. Use HH:MM")
                return None, None
            
            hour = int(parts[0])
            minute = int(parts[1])
            
            if not (0 <= hour <= 23):
                print("Invalid hour (0-23)")
                return None, None
            
            if not (0 <= minute <= 59):
                print("Invalid minutes (0-59)")
                return None, None
            
            return hour, minute
            
        except ValueError:
            print("Invalid time format. Use numbers")
            return None, None
        except KeyboardInterrupt:
            return None, None
    
    def run_interactive(self):
        self.print_header()
        
        prediction_count = 0
        
        while True:
            try:
                lat, lon, location_name = self.get_coordinates_input()
                
                if lat is None:
                    break
                
                hour, minute = self.get_time_input()
                
                print("\nCalculating prediction...")
                result = self.predict_rain(lat, lon, hour, minute)
                
                self.print_prediction(result, location_name)
                
                prediction_count += 1
                
                continue_input = input("\nMake another prediction? (Enter=Yes, n=No): ").strip().lower()
                if continue_input in ['n', 'no', 'exit']:
                    break
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue
        
        print(f"\nSession completed:")
        print(f"Predictions made: {prediction_count}")
        print(f"Thank you for using the rain predictor!")
        print("="*80)

def main():
    try:
        predictor = ConsoleRainPredictor()
        predictor.run_interactive()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()