from nicegui import ui, app
import folium
from nicegui.elements.html import Html
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Clase para predicción de lluvia
class RainPredictor:
    def __init__(self, model_path='rain_prediction_model.joblib'):
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.loaded = True
            print("✅ Modelo de predicción de lluvia cargado")
        except Exception as e:
            print(f"⚠️ Modelo de predicción no disponible: {str(e)}")
            self.model = None
            self.loaded = False
    
    def predict_rain_probability(self, lat, lon, hour=None, minute=None):
        if not self.loaded:
            return {"error": "Modelo no disponible", "category": "❌ Error"}
        
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
        elif precip_prediction > 0.5:
            category = "🌦️ Lluvia Moderada"
            probability = "Media"
        elif precip_prediction > 0.1:
            category = "🌤️ Lluvia Ligera"
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

# Instancia global del predictor
rain_predictor = RainPredictor()

@ui.page('/',title="BAM - Build Manager",response_timeout=15)
async def main_page():
    left_drawer()
    await header()
    await map_entry() 

async def header():
    with ui.header().classes('row flex h-20 items-center bg-blue-950') as head:
        ui.image('src/logo/nova_delta.png').classes('object-fit w-20 h-full')
        ui.label("Nova Delta").classes('text-2xl font-medium mb-2')

def left_drawer():
    with ui.left_drawer().classes('w-72') as drawer:
        ui.label("Parameters to show").classes('text-xl font-medium mb-2').style('font-weight: bold;')
        ui.switch('Temperature', value=True).classes('mb-2').props('left-label')
        ui.switch('Humidity', value=True).classes('mb-2').props('left-label')
        ui.switch('Pressure', value=True).classes('mb-2').props('left-label')
        ui.switch('Wind Speed', value=True).classes('mb-2').props('left-label')  
        ui.switch('Cloud Cover', value=True).classes('mb-2').props('left-label')  
        ui.switch('Precipitation', value=True).classes('mb-2').props('left-label')

async def map_entry():
    with ui.card().classes('m-4 p-4 w-full'):
        ui.label("Mapa Interactivo - Coordenadas en Consola").classes('text-xl font-bold mb-4')
        # Crear mapa con Folium
        def create_clickable_map():
            # Crear mapa centrado en el mundo
            m = folium.Map(
                location=[20, 0],  # Centro mundial
                zoom_start=2,
                tiles='OpenStreetMap'
            )
            
            # JavaScript mejorado para capturar clics
            click_script = """
            <script>
            function setupMapClickHandler() {
                // Buscar el mapa en el DOM
                setTimeout(function() {
                    // Buscar todas las instancias de mapas Leaflet
                    for (var key in window) {
                        if (key.indexOf('map_') === 0 && window[key] && window[key]._container) {
                            var map = window[key];
                            console.log('🗺️ Mapa encontrado y configurado para clics');
                            
                            // Agregar evento de clic
                            map.on('click', function(e) {
                                var lat = e.latlng.lat;
                                var lng = e.latlng.lng;
                                
                                // Imprimir en consola
                                console.log('🗺️ COORDENADAS CLICKEADAS:');
                                console.log('Latitud: ' + lat);
                                console.log('Longitud: ' + lng);
                                console.log('Coordenadas completas: [' + lat + ', ' + lng + ']');
                                
                                // Hacer predicción de lluvia si está disponible
                                fetch('/predict_rain', {
                                    method: 'POST',
                                    headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({lat: lat, lon: lng})
                                })
                                .then(response => response.json())
                                .then(data => {
                                    let popupContent = '📍 <b>Coordenadas:</b><br>' +
                                                     'Lat: ' + lat.toFixed(6) + '<br>' +
                                                     'Lng: ' + lng.toFixed(6) + '<br><br>';
                                    
                                    if (data.error) {
                                        popupContent += '❌ ' + data.error;
                                        console.log('❌ Error en predicción: ' + data.error);
                                    } else {
                                        popupContent += '<b>🌧️ Predicción de Lluvia:</b><br>' +
                                                       data.category + '<br>' +
                                                       '💧 ' + data.precipitation_mm_hr + ' mm/hr<br>' +
                                                       '📊 Probabilidad: ' + data.probability + '<br>' +
                                                       '🕒 ' + data.time;
                                        
                                        console.log('🌧️ PREDICCIÓN DE LLUVIA:');
                                        console.log('Categoría: ' + data.category);
                                        console.log('Precipitación: ' + data.precipitation_mm_hr + ' mm/hr');
                                        console.log('Probabilidad: ' + data.probability);
                                        console.log('Hora: ' + data.time);
                                    }
                                    
                                    console.log('─────────────────────────────────');
                                    
                                    // Crear popup con predicción
                                    L.popup()
                                        .setLatLng(e.latlng)
                                        .setContent(popupContent)
                                        .openOn(map);
                                })
                                .catch(error => {
                                    console.error('Error en predicción:', error);
                                    L.popup()
                                        .setLatLng(e.latlng)
                                        .setContent('📍 <b>Coordenadas:</b><br>Lat: ' + lat.toFixed(6) + '<br>Lng: ' + lng.toFixed(6) + '<br><br>❌ Error en predicción')
                                        .openOn(map);
                                });
                            });
                            
                            return; // Salir cuando encontremos el mapa
                        }
                    }
                    
                    // Si no se encuentra, intentar de nuevo
                    console.log('� Buscando mapa, reintentando...');
                    setupMapClickHandler();
                }, 1000);
            }
            
            // Inicializar cuando el DOM esté listo
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupMapClickHandler);
            } else {
                setupMapClickHandler();
            }
            </script>
            """
            
            # Agregar el script al mapa
            from folium import Element
            m.get_root().html.add_child(Element(click_script))
            
            return m
        
        # Crear y mostrar el mapa
        folium_map = create_clickable_map()
        map_html = folium_map._repr_html_()
        
        # Mostrar el mapa
        ui.html(map_html, sanitize=False).classes('w-full').style('height: 500px; border: 2px solid #ddd; border-radius: 8px;')
        
        
        # Instrucciones para el usuario
        ui.separator().classes('my-4')
        
        with ui.row().classes('w-full items-center justify-center'):
            if rain_predictor.loaded:
                ui.label("✅ Predicción de lluvia habilitada - Haz clic en el mapa").classes('text-green font-bold')
            else:
                ui.label("⚠️ Predicción de lluvia no disponible").classes('text-orange font-bold')
        
        with ui.card().classes('mt-4 p-4'):
            ui.label("🌧️ Sistema de Predicción de Lluvia").classes('text-lg font-bold')
            ui.label("• Haz clic en cualquier punto del mapa para obtener una predicción")
            ui.label("• Las predicciones aparecen en el popup y en la consola del navegador")
            ui.label("• Modelo entrenado con datos satelitales GPM IMERG")
            if rain_predictor.loaded:
                ui.label(f"• Características del modelo: {len(rain_predictor.feature_columns)}")

# Endpoint para las predicciones AJAX
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
            return {"error": "Modelo no disponible - ejecuta rain_model_robust.py"}
            
    except Exception as e:
        return {"error": f"Error en predicción: {str(e)}"}

        
       
        
        
if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title="Nova Delta Meteorology",
        favicon="src/logo/nova_delta.jpeg",
        port=8001,
        dark=True,
        reload=True,
        uvicorn_reload_excludes='.venv*',
        uvicorn_reload_dirs='src'
    )