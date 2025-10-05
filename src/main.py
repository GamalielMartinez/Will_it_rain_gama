from nicegui import ui, app
import folium
from nicegui.elements.html import Html
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
            self.model = None
            self.loaded = False
    
    def predict_rain_probability(self, lat, lon, hour=None, minute=None):
        if not self.loaded:
            return {"error": "Modelo no disponible", "category": " Error"}
        
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

        lat_factor = (lat - 25.0) / 10.0  
        lon_factor = (lon + 90.0) / 10.0  
        hour_factor = hour / 24.0  
        
        coastal_factor = 1.0 - min(abs(lat - 29.0) / 5.0, 1.0)  
        tropical_factor = max(0, 1.0 - (lat - 20.0) / 15.0)  
        
        # Valores dinámicos basados en ubicación
        base_precip = 0.1 + (coastal_factor * 0.2) + (tropical_factor * 0.15)
        location_variance = abs(lat_factor * lon_factor) * 0.3
        temporal_variance = np.sin(2 * np.pi * hour_factor) * 0.1
        
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
        
        for col in self.feature_columns:
            if col not in features:
                features[col] = default_values.get(col, 0.5)
        
        X_pred = pd.DataFrame([features])[self.feature_columns]
        X_pred_scaled = self.scaler.transform(X_pred)
        precip_prediction = max(0, self.model.predict(X_pred_scaled)[0])
        
        threshold = 0.1  
        scale_factor = 10.0  
        
        if precip_prediction <= 0.01:
            rain_probability = 0.0
        else:
            sigmoid_input = (precip_prediction - threshold) * scale_factor
            rain_probability = 1 / (1 + np.exp(-sigmoid_input))
            rain_probability = min(max(rain_probability, 0.0), 1.0)
        
        intensity_probability = min(precip_prediction / 5.0, 1.0)  # Normalizar a máximo 5 mm/hr
        
        final_probability = (rain_probability * 0.7 + intensity_probability * 0.3)
        final_probability = min(max(final_probability, 0.0), 1.0)
        
        if final_probability >= 0.9:
            category = "�️ Lluvia Muy Fuerte"
        elif final_probability >= 0.7:
            category = "🌦️ Lluvia Fuerte"
        elif final_probability >= 0.4:
            category = "🌤️ Lluvia Moderada"
        elif final_probability >= 0.1:
            category = "⛅ Lluvia Ligera"
        else:
            category = "☀️ Sin Lluvia"
        
        probability_percent = f"{final_probability * 100:.2f}%"
        
        return {
            "precipitation_mm_hr": round(precip_prediction, 4),
            "probability": probability_percent,
            "probability_decimal": round(final_probability, 4),
            "category": category,
            "coordinates": {"lat": lat, "lon": lon},
            "time": f"{hour:02d}:{minute:02d}"
        }

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
     # Controles de hora para predicción
        with ui.left_drawer():
            ui.label("Configuración de Hora para Predicción").classes('text-lg font-bold mb-2')
            
            with ui.row().classes('w-full items-center gap-4'):
                # Selector de hora
                hour_input = ui.number(
                    label="Hora (0-23)", 
                    value=datetime.now().hour,
                    min=0, 
                    max=23,
                    step=1,
                    format="%.0f"
                ).classes('w-32')
                
                # Selector de minutos
                minute_input = ui.number(
                    label="Minutos (0-59)", 
                    value=0,
                    min=0, 
                    max=59,
                    step=15,  # Intervalos de 15 minutos
                    format="%.0f"
                ).classes('w-32')
                
                # Botón para usar hora actual
                def update_current_time():
                    now = datetime.now()
                    hour_input.value = now.hour
                    minute_input.value = now.minute
                    ui.notify(f"Hora actualizada: {now.hour:02d}:{now.minute:02d}")
                
                ui.button("🕒 Usar Hora Actual", on_click=update_current_time).classes('bg-blue-500 text-white')
                
                # Mostrar hora seleccionada
                def format_selected_time():
                    return f"{int(hour_input.value):02d}:{int(minute_input.value):02d}"
                
                time_display = ui.label().classes('text-lg font-bold text-blue-600')
                
                def update_time_display():
                    time_display.text = f"Predicción para: {format_selected_time()}"
                
                # Actualizar display cuando cambien los valores
                hour_input.on('change', update_time_display)
                minute_input.on('change', update_time_display)
                update_time_display()  # Inicializar

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
                                
                                // Obtener hora seleccionada desde los inputs
                                var hourInput = document.querySelector('input[aria-label="Hora (0-23)"]');
                                var minuteInput = document.querySelector('input[aria-label="Minutos (0-59)"]');
                                var selectedHour = hourInput ? parseInt(hourInput.value) || 0 : 0;
                                var selectedMinute = minuteInput ? parseInt(minuteInput.value) || 0 : 0;
                                
                                // Hacer predicción de lluvia con hora personalizada
                                fetch('/predict_rain', {
                                    method: 'POST',
                                    headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({
                                        lat: lat, 
                                        lon: lng, 
                                        hour: selectedHour, 
                                        minute: selectedMinute
                                    })
                                })
                                .then(response => response.json())
                                .then(data => {
                                    let popupContent = ' <b>Coordenadas:</b><br>' +
                                                     'Lat: ' + lat.toFixed(6) + '<br>' +
                                                     'Lng: ' + lng.toFixed(6) + '<br><br>';
                                    
                                    if (data.error) {
                                        popupContent += ' ' + data.error;
                                        console.log(' Error en predicción: ' + data.error);
                                    } else {
                                        // Crear barra de probabilidad visual
                                        let probDecimal = data.probability_decimal || 0;
                                        let barWidth = Math.round(probDecimal * 100);
                                        let progressBar = '<div style="background: #e0e0e0; border-radius: 10px; overflow: hidden; height: 20px; margin: 5px 0;">' +
                                                         '<div style="background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #FF5722 100%); height: 100%; width: ' + barWidth + '%; transition: width 0.3s;"></div></div>';
                                        
                                        popupContent += '<b>🌧️ Predicción de Lluvia:</b><br>' +
                                                       data.category + '<br><br>' +
                                                       '<b>💧 Precipitación:</b> ' + data.precipitation_mm_hr + ' mm/hr<br>' +
                                                       '<b>📊 Probabilidad:</b> ' + data.probability + '<br>' +
                                                       progressBar 
                                        console.log('🌧️ PREDICCIÓN DETALLADA:');
                                        console.log('Ubicación: [' + lat.toFixed(6) + ', ' + lng.toFixed(6) + ']');
                                        console.log('Hora seleccionada: ' + selectedHour.toString().padStart(2, '0') + ':' + selectedMinute.toString().padStart(2, '0'));
                                        console.log('Categoría: ' + data.category);
                                        console.log('Precipitación: ' + data.precipitation_mm_hr + ' mm/hr');
                                        console.log('Probabilidad: ' + data.probability + ' (' + data.probability_decimal + ')');
                                        console.log('Tiempo modelo: ' + data.time);
                                        console.log('Confianza: ' + (data.probability_decimal * 100).toFixed(2) + '%');
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
                                        .setContent(' <b>Coordenadas:</b><br>Lat: ' + lat.toFixed(6) + '<br>Lng: ' + lng.toFixed(6) + '<br><br>❌ Error en predicción')
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
        

from fastapi import Request
import json

@app.post('/predict_rain')
async def predict_rain_endpoint(request: Request):
    """
    Endpoint para hacer predicciones de lluvia vía AJAX con hora personalizada
    """
    try:
        data = await request.json()
        lat = data.get('lat')
        lon = data.get('lon')
        hour = data.get('hour')  # Hora personalizada del usuario
        minute = data.get('minute')  # Minuto personalizado del usuario
        
        if lat is None or lon is None:
            return {"error": "Coordenadas inválidas"}
        
        # Hacer predicción con hora personalizada
        if rain_predictor.loaded:
            prediction = rain_predictor.predict_rain_probability(lat, lon, hour, minute)
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