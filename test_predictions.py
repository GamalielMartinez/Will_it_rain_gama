from console_predictor import ConsoleRainPredictor

def test_geographic_variation():
    """
    Prueba la variabilidad geográfica de las predicciones
    """
    predictor = ConsoleRainPredictor()
    
    print('🧪 PRUEBA DE VARIABILIDAD GEOGRÁFICA')
    print('='*60)
    
    locations = [
        (25.7617, -80.1918, 'Miami, FL'),
        (30.6954, -88.0399, 'Mobile, AL'), 
        (29.7604, -95.3698, 'Houston, TX'),
        (27.8006, -97.3964, 'Corpus Christi, TX'),
        (30.4213, -87.2169, 'Pensacola, FL'),
        (29.2694, -94.7847, 'Galveston, TX')
    ]
    
    print('Probando a las 14:00 (2:00 PM):')
    print('-' * 60)
    
    for lat, lon, name in locations:
        result = predictor.predict_rain(lat, lon, 14, 0)
        print(f'{name:<20} -> {result["precipitation_mm_hr"]:6.4f} mm/hr | {result["probability"]:>8} | {result["category"]}')
    
    print('\n' + '='*60)
    print('Probando Miami a diferentes horas:')
    print('-' * 60)
    
    miami_lat, miami_lon = 25.7617, -80.1918
    for hour in [6, 10, 14, 18, 22]:
        result = predictor.predict_rain(miami_lat, miami_lon, hour, 0)
        print(f'Miami a las {hour:02d}:00   -> {result["precipitation_mm_hr"]:6.4f} mm/hr | {result["probability"]:>8}')

if __name__ == "__main__":
    test_geographic_variation()