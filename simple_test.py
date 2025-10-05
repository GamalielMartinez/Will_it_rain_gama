import requests
import time
import json

print('⏳ Esperando que la aplicación esté lista...')
time.sleep(3)

try:
    payload = {
        'lat': 25.7617, 
        'lon': -80.1918, 
        'hour': 14, 
        'minute': 0
    }
    
    response = requests.post('http://localhost:8001/predict_rain', 
                           json=payload, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        print('✅ Predicción exitosa con hora personalizada:')
        print(f'   🌧️ Precipitación: {data.get("precipitation_mm_hr", "N/A")} mm/hr')
        print(f'   📊 Probabilidad: {data.get("probability", "N/A")}')
        print(f'   🏷️ Categoría: {data.get("category", "N/A")}')
        print(f'   🕒 Hora: {data.get("time", "N/A")}')
        
        # Probar con otra hora
        print('\n🔄 Probando con hora diferente (18:00)...')
        payload['hour'] = 18
        response2 = requests.post('http://localhost:8001/predict_rain', 
                                json=payload, timeout=10)
        
        if response2.status_code == 200:
            data2 = response2.json()
            print('✅ Segunda predicción exitosa:')
            print(f'   🌧️ Precipitación: {data2.get("precipitation_mm_hr", "N/A")} mm/hr')
            print(f'   📊 Probabilidad: {data2.get("probability", "N/A")}')
            print(f'   🏷️ Categoría: {data2.get("category", "N/A")}')
            print(f'   🕒 Hora: {data2.get("time", "N/A")}')
        
    else:
        print(f'❌ Error HTTP: {response.status_code}')
        print(f'Respuesta: {response.text}')
        
except Exception as e:
    print(f'❌ Error: {e}')