"""
Script para probar las predicciones con diferentes horas
"""
import requests
import json

def test_time_predictions():
    """
    Prueba las predicciones con diferentes horas para la misma ubicación
    """
    url = "http://localhost:8001/predict_rain"
    
    # Coordenadas de Miami
    lat, lon = 25.7617, -80.1918
    
    print("🧪 PRUEBA DE PREDICCIONES POR HORA")
    print("="*60)
    print(f"📍 Ubicación: Miami, FL ({lat}, {lon})")
    print("="*60)
    
    # Probar diferentes horas
    hours_to_test = [6, 9, 12, 15, 18, 21]
    
    for hour in hours_to_test:
        try:
            payload = {
                "lat": lat,
                "lon": lon,
                "hour": hour,
                "minute": 0
            }
            
            response = requests.post(url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" not in data:
                    print(f"🕒 {hour:02d}:00 -> {data['precipitation_mm_hr']:6.4f} mm/hr | {data['probability']:>8} | {data['category']}")
                else:
                    print(f"❌ {hour:02d}:00 -> Error: {data['error']}")
            else:
                print(f"❌ {hour:02d}:00 -> HTTP Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {hour:02d}:00 -> Connection Error: {e}")
        except Exception as e:
            print(f"❌ {hour:02d}:00 -> Error: {e}")
    
    print("\n" + "="*60)
    print("🌍 PRUEBA DE UBICACIONES DIFERENTES (15:00)")
    print("="*60)
    
    locations = [
        (25.7617, -80.1918, "Miami, FL"),
        (29.7604, -95.3698, "Houston, TX"),
        (30.4213, -87.2169, "Pensacola, FL")
    ]
    
    for lat, lon, name in locations:
        try:
            payload = {
                "lat": lat,
                "lon": lon,
                "hour": 15,
                "minute": 0
            }
            
            response = requests.post(url, json=payload, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" not in data:
                    print(f"{name:<20} -> {data['precipitation_mm_hr']:6.4f} mm/hr | {data['probability']:>8}")
                else:
                    print(f"{name:<20} -> Error: {data['error']}")
            else:
                print(f"{name:<20} -> HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"{name:<20} -> Error: {e}")

if __name__ == "__main__":
    test_time_predictions()