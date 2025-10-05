import pandas as pd

# Verificar el archivo CSV generado
csv_file = 'precipitation_data_3B-HHR.MS.MRG.3IMERG.20240101-S000000-E002959.0000.V07B.csv'

print("🔍 VERIFICACIÓN DEL ARCHIVO CSV")
print("=" * 50)

df = pd.read_csv(csv_file)

print(f"📄 Archivo: {csv_file}")
print(f"📊 Filas: {len(df):,}")
print(f"📊 Columnas: {list(df.columns)}")

print(f"\n🌧️  Estadísticas de precipitación:")
print(f"   - Puntos con lluvia (>0): {len(df[df['precipitation'] > 0]):,}")
print(f"   - Precipitación mínima: {df['precipitation'].min():.4f}")
print(f"   - Precipitación máxima: {df['precipitation'].max():.4f}")
print(f"   - Precipitación promedio (donde llueve): {df[df['precipitation'] > 0]['precipitation'].mean():.4f}")

print(f"\n📍 Rango de coordenadas:")
print(f"   - Latitud: {df['latitude'].min():.2f} a {df['latitude'].max():.2f}")
print(f"   - Longitud: {df['longitude'].min():.2f} a {df['longitude'].max():.2f}")

print(f"\n📋 Primeras 5 filas:")
print(df.head())

print(f"\n🌧️  Los 5 puntos con más lluvia:")
top_rain = df[df['precipitation'] > 0].nlargest(5, 'precipitation')
print(top_rain)

print(f"\n✅ El archivo CSV está listo para usar en análisis o visualización!")