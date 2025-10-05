import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import folium
from nicegui import ui, app
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
            print(f"✅ Modelo cargado exitosamente desde: {model_path}")
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
        
        # Llenar características faltantes con valores promedio
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
        
        # Convertir a probabilidad y categoría
        if precip_prediction > 2.5:
            category = "🌧️ Lluvia Fuerte"
            probability = "Alta"
            color = "red"
        elif precip_prediction > 0.5:
            category = "🌦️ Lluvia Moderada"
            probability = "Media"
            color = "orange"
        elif precip_prediction > 0.1:
            category = "🌤️ Lluvia Ligera"
            probability = "Baja"
            color = "yellow"
        else:
            category = "☀️ Sin Lluvia"
            probability = "Muy Baja"
            color = "green"
        
        return {
            "precipitation_mm_hr": round(precip_prediction, 4),
            "probability": probability,
            "category": category,
            "color": color,
            "coordinates": {"lat": lat, "lon": lon},
            "time": f"{hour:02d}:{minute:02d}",
            "location_type": self.get_location_type(lat, lon)
        }
    
    def get_location_type(self, lat, lon):
        """
        Determina el tipo de ubicación basado en las coordenadas
        """
        # Golfo de México aproximado
        if 18 <= lat <= 31 and -98 <= lon <= -80:
            if lat < 26:
                return "Golfo Sur"
            elif lat > 29:
                return "Golfo Norte"
            else:
                return "Golfo Centro"
        else:
            return "Fuera del Golfo"

def create_rain_prediction_app():
    """
    Crea una aplicación web interactiva para predicción de lluvia
    """
    # Cargar el predictor
    predictor = RainPredictor()
    
    if predictor.model is None:
        ui.label("❌ Error: No se pudo cargar el modelo de predicción")
        return
    
    # Variables para almacenar resultados
    prediction_result = {"data": None}
    
    def predict_for_coordinates():
        """
        Realiza predicción para las coordenadas especificadas
        """
        try:
            lat = float(lat_input.value)
            lon = float(lon_input.value)
            hour = int(hour_input.value) if hour_input.value else None
            minute = int(minute_input.value) if minute_input.value else None
            
            # Validar coordenadas
            if not (-90 <= lat <= 90):
                result_card.clear()
                with result_card:
                    ui.label("❌ Latitud debe estar entre -90 y 90", style="color: red")
                return
            
            if not (-180 <= lon <= 180):
                result_card.clear()
                with result_card:
                    ui.label("❌ Longitud debe estar entre -180 y 180", style="color: red")
                return
            
            # Hacer predicción
            result = predictor.predict_rain_probability(lat, lon, hour, minute)
            prediction_result["data"] = result
            
            # Mostrar resultado
            result_card.clear()
            with result_card:
                ui.label(f"🎯 Predicción para ({lat}, {lon})", style="font-size: 18px; font-weight: bold")
                ui.separator()
                
                with ui.row():
                    with ui.card().style("padding: 15px; margin: 5px"):
                        ui.label(result["category"], style="font-size: 16px")
                        ui.label(f"Intensidad: {result['precipitation_mm_hr']} mm/hr")
                        ui.label(f"Probabilidad: {result['probability']}")
                        ui.label(f"Hora: {result['time']}")
                        ui.label(f"Región: {result['location_type']}")
            
            # Actualizar mapa
            update_map()
            
        except ValueError:
            result_card.clear()
            with result_card:
                ui.label("❌ Por favor ingresa valores numéricos válidos", style="color: red")
        except Exception as e:
            result_card.clear()
            with result_card:
                ui.label(f"❌ Error: {str(e)}", style="color: red")
    
    def update_map():
        """
        Actualiza el mapa con la predicción
        """
        if prediction_result["data"] is None:
            return
        
        result = prediction_result["data"]
        lat = result["coordinates"]["lat"]
        lon = result["coordinates"]["lon"]
        
        # Crear mapa centrado en la predicción
        m = folium.Map(location=[lat, lon], zoom_start=8)
        
        # Añadir marcador con la predicción
        popup_text = f"""
        <b>{result['category']}</b><br>
        Precipitación: {result['precipitation_mm_hr']} mm/hr<br>
        Probabilidad: {result['probability']}<br>
        Hora: {result['time']}<br>
        Región: {result['location_type']}
        """
        
        folium.Marker(
            [lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=result['color'], icon='cloud')
        ).add_to(m)
        
        # Añadir círculo para mostrar área de influencia
        folium.Circle(
            [lat, lon],
            radius=10000,  # 10 km
            popup=f"Área de predicción",
            color=result['color'],
            fill=True,
            opacity=0.3
        ).add_to(m)
        
        map_container.clear()
        with map_container:
            ui.html(m._repr_html_()).style("height: 400px")
    
    def on_map_click(event):
        """
        Maneja clics en el mapa
        """
        lat = event.latlng.lat
        lon = event.latlng.lng
        lat_input.value = str(round(lat, 4))
        lon_input.value = str(round(lon, 4))
        predict_for_coordinates()
    
    # Interfaz de usuario
    ui.label("🌧️ Predictor de Lluvia con Machine Learning", 
             style="font-size: 24px; font-weight: bold; text-align: center")
    
    ui.separator()
    
    with ui.card().style("padding: 20px; margin: 10px"):
        ui.label("📍 Coordenadas para Predicción", style="font-size: 18px; font-weight: bold")
        
        with ui.row():
            lat_input = ui.number("Latitud", value=29.0, format="%.4f", min=-90, max=90)
            lon_input = ui.number("Longitud", value=-94.0, format="%.4f", min=-180, max=180)
        
        with ui.row():
            hour_input = ui.number("Hora (0-23)", value=datetime.now().hour, min=0, max=23)
            minute_input = ui.number("Minuto (0-59)", value=datetime.now().minute, min=0, max=59)
        
        ui.button("🔮 Predecir Lluvia", on_click=predict_for_coordinates)
        
        # Ejemplos de coordenadas
        ui.label("📌 Coordenadas de ejemplo:", style="margin-top: 10px")
        with ui.row():
            ui.button("Houston", on_click=lambda: [setattr(lat_input, 'value', 29.7604), setattr(lon_input, 'value', -95.3698), predict_for_coordinates()])
            ui.button("Miami", on_click=lambda: [setattr(lat_input, 'value', 25.7617), setattr(lon_input, 'value', -80.1918), predict_for_coordinates()])
            ui.button("New Orleans", on_click=lambda: [setattr(lat_input, 'value', 29.9511), setattr(lon_input, 'value', -90.0715), predict_for_coordinates()])
    
    # Resultados
    result_card = ui.card().style("padding: 20px; margin: 10px")
    with result_card:
        ui.label("👆 Ingresa coordenadas y haz clic en 'Predecir Lluvia'")
    
    # Mapa
    with ui.card().style("padding: 20px; margin: 10px"):
        ui.label("🗺️ Mapa Interactivo", style="font-size: 18px; font-weight: bold")
        map_container = ui.html()
        
        # Mapa inicial
        initial_map = folium.Map(location=[29.0, -94.0], zoom_start=6)
        with map_container:
            ui.html(initial_map._repr_html_()).style("height: 400px")

def main():
    """
    Función principal que ejecuta la aplicación
    """
    print("🌧️ Iniciando aplicación de predicción de lluvia...")
    
    # Verificar que el modelo existe
    import os
    if not os.path.exists('rain_prediction_model.joblib'):
        print("❌ Error: No se encontró el modelo entrenado.")
        print("🔄 Ejecuta primero 'rain_model_robust.py' para entrenar el modelo.")
        return
    
    create_rain_prediction_app()

if __name__ == "__main__":
    main()