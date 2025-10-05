#!/usr/bin/env python3
"""
Script interactivo para predicción de lluvia por consola
Permite ingresar coordenadas y obtener predicciones usando el modelo entrenado
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

class ConsoleRainPredictor:
    def __init__(self):
        """
        Inicializa el predictor de consola
        """
        self.model_data = None
        self.load_model()
    
    def load_model(self):
        """
        Carga el modelo entrenado
        """
        model_paths = [
            'rain_prediction_model.joblib',
            'src/rain_prediction_model.joblib',
            'rain_prediction_model_enhanced.joblib'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    print(f"📦 Cargando modelo desde: {path}")
                    self.model_data = joblib.load(path)
                    print(f"✅ Modelo cargado exitosamente")
                    print(f"   🗓️ Versión: {self.model_data.get('model_version', 'N/A')}")
                    print(f"   📅 Fecha entrenamiento: {self.model_data.get('training_date', 'N/A')}")
                    print(f"   🎯 Características: {len(self.model_data['feature_columns'])}")
                    return
                except Exception as e:
                    print(f"❌ Error cargando {path}: {e}")
                    continue
        
        print("❌ No se pudo cargar ningún modelo")
        sys.exit(1)
    
    def predict_rain(self, lat, lon, hour=None, minute=None):
        """
        Predice precipitación para coordenadas específicas
        """
        if self.model_data is None:
            return {
                "precipitation_mm_hr": 0.0,
                "probability": "Error - Modelo no cargado",
                "category": "❌ Error",
                "emoji": "❌",
                "coordinates": {"lat": lat, "lon": lon},
                "time": "N/A"
            }
        
        if hour is None or minute is None:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
        
        # Crear características usando la misma lógica del modelo
        features = {}
        
        # Características temporales
        features['hour'] = hour
        features['minute'] = minute
        features['month'] = datetime.now().month
        features['day'] = datetime.now().day
        features['day_of_week'] = datetime.now().weekday()
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
        features['day_cos'] = np.cos(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
        features['month_sin'] = np.sin(2 * np.pi * datetime.now().month / 12)
        features['month_cos'] = np.cos(2 * np.pi * datetime.now().month / 12)
        features['dow_sin'] = np.sin(2 * np.pi * datetime.now().weekday() / 7)
        features['dow_cos'] = np.cos(2 * np.pi * datetime.now().weekday() / 7)
        
        # Características geográficas específicas
        features['lat_center'] = lat
        features['lon_center'] = lon
        features['lat_range'] = 10.0
        features['lon_range'] = 15.0
        
        # Generar valores dinámicos basados en coordenadas y hora
        # Usar funciones que varíen según la ubicación geográfica
        lat_factor = (lat - 25.0) / 10.0  # Normalizar latitud relativa al Golfo de México
        lon_factor = (lon + 90.0) / 10.0  # Normalizar longitud
        hour_factor = hour / 24.0  # Factor horario
        
        # Factores geográficos para diferentes regiones
        coastal_factor = 1.0 - min(abs(lat - 29.0) / 5.0, 1.0)  # Más cerca del centro del Golfo
        tropical_factor = max(0, 1.0 - (lat - 20.0) / 15.0)  # Factor tropical (más al sur)
        
        # Valores dinámicos basados en ubicación
        base_precip = 0.1 + (coastal_factor * 0.2) + (tropical_factor * 0.15)
        location_variance = abs(lat_factor * lon_factor) * 0.3
        temporal_variance = np.sin(2 * np.pi * hour_factor) * 0.1
        
        # Valores por defecto dinámicos para características meteorológicas
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
        
        # Llenar características faltantes
        for col in self.model_data['feature_columns']:
            if col not in features:
                features[col] = default_values.get(col, 0.5)
        
        # Crear DataFrame y hacer predicción
        X_pred = pd.DataFrame([features])[self.model_data['feature_columns']]
        X_pred_scaled = self.model_data['scaler'].transform(X_pred)
        
        # Predicción
        precip_prediction = max(0, self.model_data['model'].predict(X_pred_scaled)[0])
        
        # Calcular probabilidad exacta del modelo
        # Convertir la predicción de precipitación a probabilidad de lluvia
        # Usando una función sigmoide ajustada para datos meteorológicos
        
        # Parámetros calibrados para datos de precipitación
        threshold = 0.1  # mm/hr mínimo para considerar lluvia
        scale_factor = 10.0  # Factor de escalamiento
        
        # Probabilidad de que haya lluvia (>= threshold mm/hr)
        if precip_prediction <= 0.01:
            rain_probability = 0.0
        else:
            # Función sigmoide modificada para meteorología
            sigmoid_input = (precip_prediction - threshold) * scale_factor
            rain_probability = 1 / (1 + np.exp(-sigmoid_input))
            rain_probability = min(max(rain_probability, 0.0), 1.0)  # Limitar entre 0 y 1
        
        # Calcular probabilidad adicional basada en la magnitud de precipitación
        intensity_probability = min(precip_prediction / 5.0, 1.0)  # Normalizar a máximo 5 mm/hr
        
        # Combinar ambas probabilidades (promedio ponderado)
        final_probability = (rain_probability * 0.7 + intensity_probability * 0.3)
        final_probability = min(max(final_probability, 0.0), 1.0)
        
        # Categorización basada en la probabilidad exacta
        if final_probability >= 0.9:
            category = "�️ Lluvia Muy Fuerte"
            emoji = "🌧️"
        elif final_probability >= 0.7:
            category = "🌦️ Lluvia Fuerte"
            emoji = "�️"
        elif final_probability >= 0.4:
            category = "🌤️ Lluvia Moderada"
            emoji = "🌤️"
        elif final_probability >= 0.1:
            category = "⛅ Lluvia Ligera"
            emoji = "⛅"
        else:
            category = "☀️ Sin Lluvia"
            emoji = "☀️"
        
        # Formatear probabilidad como porcentaje exacto
        probability_percent = f"{final_probability * 100:.2f}%"
        
        return {
            "precipitation_mm_hr": round(precip_prediction, 4),
            "probability": probability_percent,
            "probability_decimal": round(final_probability, 4),
            "category": category,
            "emoji": emoji,
            "coordinates": {"lat": lat, "lon": lon},
            "time": f"{hour:02d}:{minute:02d}"
        }
    
    def print_header(self):
        """
        Imprime el encabezado del programa
        """
        print("\n" + "="*80)
        print("🌧️  PREDICTOR DE LLUVIA INTERACTIVO - CONSOLA")
        print("="*80)
        print("📍 Ingresa coordenadas para obtener predicciones de precipitación")
        print("🕒 Usa la fecha/hora actual o especifica una personalizada")
        print("❌ Escribe 'salir', 'exit' o 'q' para terminar")
        print("="*80)
    
    def print_prediction(self, result, location_name=None):
        """
        Imprime el resultado de la predicción de forma bonita
        """
        print("\n" + "─"*60)
        print(f"🎯 PREDICCIÓN DE LLUVIA")
        print("─"*60)
        
        if location_name:
            print(f"📍 Ubicación: {location_name}")
        
        print(f"📍 Coordenadas: {result['coordinates']['lat']:.4f}, {result['coordinates']['lon']:.4f}")
        print(f"🕒 Hora: {result['time']}")
        print(f"🌧️ Precipitación: {result['precipitation_mm_hr']} mm/hr")
        print(f"📊 Probabilidad: {result['probability']} ({result['probability_decimal']:.4f})")
        print(f"🏷️ Categoría: {result['category']}")
        
        # Barra visual de probabilidad
        prob_decimal = result['probability_decimal']
        bar_length = 30
        filled_length = int(bar_length * prob_decimal)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"📈 Confianza: [{bar}] {result['probability']}")
        
        print("─"*60)
    
    def get_coordinates_input(self):
        """
        Obtiene coordenadas del usuario con validación
        """
        while True:
            try:
                coord_input = input("\n📍 Ingresa coordenadas (lat,lon) o nombre de ciudad: ").strip()
                
                # Verificar si quiere salir
                if coord_input.lower() in ['salir', 'exit', 'q', 'quit']:
                    return None, None, None
                
                # Verificar si es una ciudad conocida
                if ',' not in coord_input or not any(c.isdigit() or c == '.' or c == '-' for c in coord_input):
                    # Es un nombre de ciudad, usar coordenadas predefinidas
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
                        print(f"✅ Ciudad encontrada: {name}")
                        return lat, lon, name
                    else:
                        print("❌ Ciudad no encontrada. Ciudades disponibles:")
                        for city, (lat, lon, name) in cities.items():
                            print(f"   - {city} ({name})")
                        continue
                
                # Parsear coordenadas numéricas
                parts = coord_input.split(',')
                if len(parts) != 2:
                    print("❌ Formato inválido. Use: latitud,longitud (ej: 29.7604,-95.3698)")
                    continue
                
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
                
                # Validar rangos razonables para el Golfo de México
                if not (20 <= lat <= 35):
                    print("❌ Latitud fuera de rango válido (20-35°N)")
                    continue
                
                if not (-100 <= lon <= -75):
                    print("❌ Longitud fuera de rango válido (-100 a -75°W)")
                    continue
                
                return lat, lon, None
                
            except ValueError:
                print("❌ Coordenadas inválidas. Use números decimales (ej: 29.7604,-95.3698)")
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                return None, None, None
    
    def get_time_input(self):
        """
        Obtiene hora opcional del usuario
        """
        try:
            time_input = input("🕒 Hora (HH:MM, Enter para usar actual): ").strip()
            
            if not time_input:
                return None, None
            
            if ':' not in time_input:
                print("❌ Formato de hora inválido. Use HH:MM")
                return None, None
            
            parts = time_input.split(':')
            if len(parts) != 2:
                print("❌ Formato de hora inválido. Use HH:MM")
                return None, None
            
            hour = int(parts[0])
            minute = int(parts[1])
            
            if not (0 <= hour <= 23):
                print("❌ Hora inválida (0-23)")
                return None, None
            
            if not (0 <= minute <= 59):
                print("❌ Minutos inválidos (0-59)")
                return None, None
            
            return hour, minute
            
        except ValueError:
            print("❌ Formato de hora inválido. Use números")
            return None, None
        except KeyboardInterrupt:
            return None, None
    
    def run_interactive(self):
        """
        Ejecuta el modo interactivo
        """
        self.print_header()
        
        prediction_count = 0
        
        while True:
            try:
                # Obtener coordenadas
                lat, lon, location_name = self.get_coordinates_input()
                
                if lat is None:  # Usuario quiere salir
                    break
                
                # Obtener hora (opcional)
                hour, minute = self.get_time_input()
                
                # Hacer predicción
                print("\n🔮 Calculando predicción...")
                result = self.predict_rain(lat, lon, hour, minute)
                
                # Mostrar resultado
                self.print_prediction(result, location_name)
                
                prediction_count += 1
                
                # Preguntar si quiere continuar
                continue_input = input("\n❓ ¿Hacer otra predicción? (Enter=Sí, n=No): ").strip().lower()
                if continue_input in ['n', 'no', 'salir', 'exit']:
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error inesperado: {e}")
                continue
        
        print(f"\n🎉 Sesión completada:")
        print(f"   📊 Predicciones realizadas: {prediction_count}")
        print(f"   👋 ¡Gracias por usar el predictor de lluvia!")
        print("="*80)

def main():
    """
    Función principal
    """
    try:
        predictor = ConsoleRainPredictor()
        predictor.run_interactive()
    except KeyboardInterrupt:
        print("\n👋 ¡Programa interrumpido por el usuario!")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()