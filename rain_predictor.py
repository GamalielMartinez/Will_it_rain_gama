import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RainPredictor:
    def __init__(self, model_path='rain_prediction_model.joblib'):
        """
        Inicializa el predictor de lluvia cargando el modelo entrenado
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            print(f"✅ Modelo cargado exitosamente")
            print(f"🔍 Características del modelo: {len(self.feature_columns)}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            self.model = None
    
    def predict_rain_probability(self, lat, lon, hour=None, minute=None):
        """
        Predice la probabilidad de lluvia para coordenadas específicas
        """
        if self.model is None:
            return {"error": "Modelo no disponible"}
        
        # Usar hora actual si no se especifica
        if hour is None or minute is None:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
        
        # Crear características para la predicción
        features = {}
        
        # Características temporales
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
        
        # Características geográficas
        if 'lat_center' in self.feature_columns:
            features['lat_center'] = lat
        if 'lon_center' in self.feature_columns:
            features['lon_center'] = lon
        if 'lat_range' in self.feature_columns:
            features['lat_range'] = 10.0
        if 'lon_range' in self.feature_columns:
            features['lon_range'] = 15.0
        
        # Llenar características faltantes con valores promedio basados en el entrenamiento
        default_values = {
            'precip_mean': 0.3,
            'precip_max': 1.0,
            'precip_std': 0.5,
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
        
        for col in self.feature_columns:
            if col not in features:
                features[col] = default_values.get(col, 0.5)
        
        # Crear DataFrame y hacer predicción
        X_pred = pd.DataFrame([features])[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Predicción (precipitación en mm/hr)
        precip_prediction = max(0, self.model.predict(X_pred_scaled)[0])
        
        # Convertir a categorías interpretables
        if precip_prediction > 2.5:
            category = "Lluvia Fuerte"
            probability = "Alta (>75%)"
            emoji = "🌧️"
            intensity = "Fuerte"
        elif precip_prediction > 0.5:
            category = "Lluvia Moderada"
            probability = "Media (50-75%)"
            emoji = "🌦️"
            intensity = "Moderada"
        elif precip_prediction > 0.1:
            category = "Lluvia Ligera"
            probability = "Baja (25-50%)"
            emoji = "🌤️"
            intensity = "Ligera"
        else:
            category = "Sin Lluvia"
            probability = "Muy Baja (<25%)"
            emoji = "☀️"
            intensity = "Ninguna"
        
        return {
            "precipitation_mm_hr": round(precip_prediction, 4),
            "probability": probability,
            "category": category,
            "emoji": emoji,
            "intensity": intensity,
            "coordinates": {"lat": lat, "lon": lon},
            "time": f"{hour:02d}:{minute:02d}",
            "location_type": self.get_location_type(lat, lon)
        }
    
    def get_location_type(self, lat, lon):
        """
        Determina el tipo de ubicación basado en las coordenadas
        """
        # Golfo de México y zonas costeras
        if 18 <= lat <= 31 and -98 <= lon <= -80:
            if lat < 24:
                return "Golfo Sur (México/Caribe)"
            elif lat > 29:
                return "Golfo Norte (Estados Unidos)"
            else:
                return "Golfo Centro"
        elif 25 <= lat <= 35 and -85 <= lon <= -75:
            return "Costa Atlántica"
        elif 25 <= lat <= 35 and -105 <= lon <= -95:
            return "Texas/Louisiana"
        else:
            return "Fuera de la región del modelo"
    
    def predict_multiple_locations(self, locations):
        """
        Predice para múltiples ubicaciones
        """
        results = []
        for location in locations:
            lat, lon, name = location
            prediction = self.predict_rain_probability(lat, lon)
            prediction["location_name"] = name
            results.append(prediction)
        return results

def main():
    """
    Función principal para hacer predicciones interactivas
    """
    print("🌧️ PREDICTOR DE LLUVIA CON MACHINE LEARNING")
    print("=" * 60)
    
    # Cargar el predictor
    predictor = RainPredictor()
    
    if predictor.model is None:
        print("❌ No se pudo cargar el modelo. Asegúrate de que 'rain_prediction_model.joblib' existe.")
        print("💡 Ejecuta 'rain_model_robust.py' primero para entrenar el modelo.")
        return
    
    # Predicciones para ubicaciones de ejemplo
    print("\n🎯 PREDICCIONES PARA UBICACIONES POPULARES:")
    print("-" * 60)
    
    example_locations = [
        (29.7604, -95.3698, "Houston, TX"),
        (25.7617, -80.1918, "Miami, FL"),
        (29.9511, -90.0715, "New Orleans, LA"),
        (27.8006, -97.3964, "Corpus Christi, TX"),
        (30.6944, -88.0431, "Mobile, AL"),
        (28.5383, -81.3792, "Orlando, FL"),
        (32.0835, -81.0998, "Savannah, GA")
    ]
    
    results = predictor.predict_multiple_locations(example_locations)
    
    for result in results:
        print(f"\n📍 {result['location_name']}")
        print(f"   {result['emoji']} {result['category']}")
        print(f"   💧 Precipitación: {result['precipitation_mm_hr']} mm/hr")
        print(f"   📊 Probabilidad: {result['probability']}")
        print(f"   🕒 Hora: {result['time']}")
        print(f"   🌍 Región: {result['location_type']}")
    
    # Modo interactivo
    print(f"\n🔮 MODO INTERACTIVO")
    print("-" * 60)
    print("Ingresa coordenadas para obtener predicciones personalizadas.")
    print("(Presiona Enter sin datos para salir)")
    
    while True:
        try:
            print(f"\n📍 Nueva predicción:")
            
            lat_input = input("Latitud (-90 a 90): ").strip()
            if not lat_input:
                break
            
            lon_input = input("Longitud (-180 a 180): ").strip()
            if not lon_input:
                break
            
            lat = float(lat_input)
            lon = float(lon_input)
            
            # Validar coordenadas
            if not (-90 <= lat <= 90):
                print("❌ Latitud debe estar entre -90 y 90")
                continue
            
            if not (-180 <= lon <= 180):
                print("❌ Longitud debe estar entre -180 y 180")
                continue
            
            # Opción de especificar hora
            hour_input = input("Hora (0-23, Enter para hora actual): ").strip()
            minute_input = input("Minuto (0-59, Enter para minuto actual): ").strip()
            
            hour = int(hour_input) if hour_input else None
            minute = int(minute_input) if minute_input else None
            
            # Hacer predicción
            result = predictor.predict_rain_probability(lat, lon, hour, minute)
            
            # Mostrar resultado
            print(f"\n🎯 RESULTADO DE LA PREDICCIÓN:")
            print(f"   📍 Coordenadas: ({lat}, {lon})")
            print(f"   {result['emoji']} {result['category']}")
            print(f"   💧 Precipitación estimada: {result['precipitation_mm_hr']} mm/hr")
            print(f"   📊 Probabilidad: {result['probability']}")
            print(f"   🕒 Hora de predicción: {result['time']}")
            print(f"   🌍 Tipo de región: {result['location_type']}")
            
            # Interpretación adicional
            if result['precipitation_mm_hr'] > 0.5:
                print(f"   ⚠️ Recomendación: Considera llevar paraguas")
            elif result['precipitation_mm_hr'] > 0.1:
                print(f"   💡 Recomendación: Posible llovizna ligera")
            else:
                print(f"   ✅ Recomendación: Tiempo seco probable")
        
        except ValueError:
            print("❌ Error: Ingresa valores numéricos válidos")
        except KeyboardInterrupt:
            print(f"\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")
    
    print(f"\n✅ Sesión de predicción terminada.")

if __name__ == "__main__":
    main()