import folium
from nicegui import ui
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RainPredictor:
    def __init__(self, model_path='rain_prediction_model.joblib'):
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.loaded = True
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)}")
            self.model = None
            self.loaded = False
    
    def predict_rain_probability(self, lat, lon, hour=None, minute=None):
        if not self.loaded:
            return {"error": "Modelo no disponible", "category": "Error"}
        
        if hour is None or minute is None:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
        
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
        
        if 'lat_center' in self.feature_columns:
            features['lat_center'] = lat
        if 'lon_center' in self.feature_columns:
            features['lon_center'] = lon
        if 'lat_range' in self.feature_columns:
            features['lat_range'] = 10.0
        if 'lon_range' in self.feature_columns:
            features['lon_range'] = 15.0
        
        default_values = {
            'precip_mean': 0.3, 'precip_max': 1.0, 'precip_std': 0.5,
            'precip_median': 0.2, 'precip_p75': 0.8, 'precip_p25': 0.1,
            'rain_coverage': 0.3, 'light_rain_ratio': 0.2, 'moderate_rain_ratio': 0.1,
            'heavy_rain_ratio': 0.05, 'prob_liquid_mean': 0.5, 'prob_liquid_max': 0.8,
            'quality_mean': 0.7, 'precip_trend_3': 0.4, 'precip_trend_6': 0.3
        }
        
        for col in self.feature_columns:
            if col not in features:
                features[col] = default_values.get(col, 0.5)
        
        X_pred = pd.DataFrame([features])[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred)
        precip_prediction = max(0, self.model.predict(X_pred_scaled)[0])
        
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
            "time": f"{hour:02d}:{minute:02d}"
        }

# Variable global para el predictor
rain_predictor = RainPredictor()

def gmp_to_csv():
    """Función original de conversión GPM a CSV"""
    print("Esta función convertirá archivos GPM HDF5 a CSV")
    # Mantener la función original si es necesaria

def map_entry():
    """
    Crea el mapa interactivo con predicción de lluvia
    """
    # Crear el mapa base
    m = folium.Map(location=[29.0, -94.0], zoom_start=6)
    
    # Añadir algunos marcadores de ejemplo con predicciones
    example_locations = [
        (29.7604, -95.3698, "Houston, TX"),
        (25.7617, -80.1918, "Miami, FL"),
        (29.9511, -90.0715, "New Orleans, LA"),
        (27.8006, -97.3964, "Corpus Christi, TX")
    ]
    
    for lat, lon, name in example_locations:
        if rain_predictor.loaded:
            prediction = rain_predictor.predict_rain_probability(lat, lon)
            popup_text = f"""
            <b>{name}</b><br>
            {prediction['category']}<br>
            Precipitación: {prediction['precipitation_mm_hr']} mm/hr<br>
            Probabilidad: {prediction['probability']}<br>
            Hora: {prediction['time']}
            """
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=prediction['color'], icon='cloud')
            ).add_to(m)
        else:
            folium.Marker(
                [lat, lon],
                popup=f"<b>{name}</b><br>Modelo no disponible",
                icon=folium.Icon(color='gray', icon='info-sign')
            ).add_to(m)
    
    # JavaScript para capturar clics y hacer predicciones
    click_js = f"""
    function onMapClick(e) {{
        const lat = e.latlng.lat.toFixed(4);
        const lon = e.latlng.lng.toFixed(4);
        
        // Enviar coordenadas al backend de Python
        fetch('/predict_rain', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{lat: parseFloat(lat), lon: parseFloat(lon)}})
        }})
        .then(response => response.json())
        .then(data => {{
            const coords = `(${{lat}}, ${{lon}})`;
            let message = `📍 Coordenadas: ${{coords}}\\n`;
            
            if (data.error) {{
                message += `❌ ${{data.error}}`;
            }} else {{
                message += `${{data.category}}\\n`;
                message += `💧 Precipitación: ${{data.precipitation_mm_hr}} mm/hr\\n`;
                message += `📊 Probabilidad: ${{data.probability}}\\n`;
                message += `🕒 Hora: ${{data.time}}`;
            }}
            
            console.log(message);
            alert(message);
        }})
        .catch(error => {{
            console.error('Error:', error);
            alert(`📍 Coordenadas: ${{coords}}\\n❌ Error en la predicción`);
        }});
    }}
    
    // Añadir listener de clics al mapa
    {m.get_name()}.on('click', onMapClick);
    """
    
    # Añadir el JavaScript al mapa
    m.get_root().html.add_child(folium.Element(f'<script>{click_js}</script>'))
    
    # Crear la interfaz de usuario
    with ui.card():
        ui.label("🌧️ Mapa de Predicción de Lluvia con Machine Learning").classes('text-h4')
        ui.label("Haz clic en cualquier lugar del mapa para obtener una predicción de lluvia")
        
        if rain_predictor.loaded:
            ui.label("✅ Modelo de predicción cargado correctamente").classes('text-green')
        else:
            ui.label("❌ Modelo de predicción no disponible").classes('text-red')
            ui.label("💡 Ejecuta 'rain_model_robust.py' para entrenar el modelo")
    
    with ui.card():
        ui.label("🗺️ Mapa Interactivo:")
        # Mostrar el mapa
        ui.html(m._repr_html_(), sanitize=False).classes('w-full h-96')
    
    with ui.card():
        ui.label("📊 Información del Modelo:").classes('text-h6')
        if rain_predictor.loaded:
            ui.label(f"• Características: {len(rain_predictor.feature_columns)}")
            ui.label("• Tipo: Random Forest Regressor")
            ui.label("• Entrenado con datos GPM IMERG de precipitación")
            ui.label("• Región: Golfo de México y zonas costeras")
        
        ui.label("🎯 Leyenda:").classes('text-h6')
        ui.label("🟢 Verde: Sin lluvia (probabilidad muy baja)")
        ui.label("🟡 Amarillo: Lluvia ligera (probabilidad baja)")
        ui.label("🟠 Naranja: Lluvia moderada (probabilidad media)")
        ui.label("🔴 Rojo: Lluvia fuerte (probabilidad alta)")

# Endpoint para las predicciones AJAX
from nicegui import app
from fastapi import Request
import json

@app.post('/predict_rain')
async def predict_rain_endpoint(request: Request):
    """
    Endpoint para hacer predicciones de lluvia vía AJAX
    """
    try:
        data = await request.json()
        lat = data.get('lat')
        lon = data.get('lon')
        
        if lat is None or lon is None:
            return {"error": "Coordenadas inválidas"}
        
        # Hacer predicción
        if rain_predictor.loaded:
            prediction = rain_predictor.predict_rain_probability(lat, lon)
            return prediction
        else:
            return {"error": "Modelo no disponible"}
            
    except Exception as e:
        return {"error": f"Error en predicción: {str(e)}"}

# Función principal
def main():
    """
    Función principal que ejecuta la aplicación
    """
    print("🌧️ Iniciando aplicación de predicción de lluvia...")
    
    # Verificar que el modelo existe
    import os
    if not os.path.exists('rain_prediction_model.joblib'):
        print("⚠️ Advertencia: No se encontró el modelo entrenado.")
        print("🔄 La aplicación funcionará con funciones limitadas.")
        print("💡 Ejecuta 'rain_model_robust.py' para entrenar el modelo.")
    
    # Crear la aplicación
    map_entry()

if __name__ == "__main__":
    main()