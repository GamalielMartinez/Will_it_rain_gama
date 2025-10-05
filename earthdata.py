
import earthaccess
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import h5py

load_dotenv()

user: str = os.environ['EARTHDATA_USERNAME']
password: str = os.environ['EARTHDATA_PASSWORD']

print(f"Intentando autenticar con usuario: {user}")

# Paso 1: Autenticación
auth = earthaccess.login(strategy='environment')

# Paso 2: Buscar un dataset específico de precipitación (GPM - Global Precipitation Measurement)
print("\n" + "=" * 70)
print("BÚSQUEDA DE COLECCIONES DE DATOS")
print("=" * 70)

# Buscar la colección GPM IMERG (un producto popular de precipitación)
collections = earthaccess.search_datasets(
    keyword='GPM IMERG',
    cloud_hosted=True
)

print(f"\nSe encontraron {len(collections)} colecciones relacionadas con GPM IMERG.")

# Mostrar información de las primeras colecciones
for i, collection in enumerate(collections[:3]):
    print(f"\n--- Colección {i+1} ---")
    print(f"Título: {collection['umm']['EntryTitle']}")
    print(f"Short Name: {collection['umm']['ShortName']}")
    print(f"Version: {collection['umm'].get('Version', 'N/A')}")
    print(f"Resumen: {collection['umm']['Abstract'][:200]}...")

# Paso 3: Buscar granules (archivos individuales) de precipitación
print("\n" + "=" * 70)
print("BÚSQUEDA DE ARCHIVOS (GRANULES)")
print("=" * 70)

# Buscar datos de precipitación GPM para una región y período específicos
granules = earthaccess.search_data(
    short_name='GPM_3IMERGHH',  # GPM IMERG Half Hourly
    cloud_hosted=True,
    bounding_box=(-100.0, 25.0, -85.0, 35.0),  # Región del Golfo de México
    temporal=("2024-01-01", "2024-01-02"),  # Solo 1 día para ejemplo
    count=10  # Limitar a 10 resultados
)

print(f"\nSe encontraron {len(granules)} archivos de precipitación.")

# Mostrar información detallada de los primeros archivos
for i, granule in enumerate(granules[:5]):
    print(f"\n--- Archivo {i+1} ---")
    print(f"Nombre: {granule['umm']['GranuleUR']}")
    print(f"Tamaño: {granule['umm']['DataGranule']['ArchiveAndDistributionInformation'][0].get('Size', 'N/A')} MB")
    print(f"Fecha: {granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']}")
    
# Paso 4: Descargar algunos archivos (comentado por defecto para no descargar automáticamente)
print("\n" + "=" * 70)
print("DESCARGA DE ARCHIVOS")
print("=" * 70)
print("\nPara descargar los archivos, descomenta las siguientes líneas:")
print("# downloaded_files = earthaccess.download(granules[:2], './data')")
print("# print('Archivos descargados:', downloaded_files)")

# Descomenta estas líneas para descargar:
downloaded_files = earthaccess.download(granules[:2], './data')
print("\nArchivos descargados:")
for file in downloaded_files:
    print(f"  - {file}")


# Convertir archivos HDF5 a CSV con coordenadas y precipitación
print("\n" + "=" * 70)
print("CONVERSIÓN A CSV")
print("=" * 70)

def convert_gpm_to_csv_with_coordinates(hdf5_file, output_file):
    """
    Convierte archivo GPM IMERG a CSV con coordenadas y precipitación
    """
    print(f"\n🔄 Procesando archivo: {os.path.basename(hdf5_file)}")
    
    try:
        with h5py.File(hdf5_file, 'r') as f:
            # Leer coordenadas
            print("📍 Leyendo coordenadas...")
            latitudes = f['Grid/lat'][:]
            longitudes = f['Grid/lon'][:]
            
            print(f"   - Latitudes: {len(latitudes)} valores ({latitudes.min():.2f} a {latitudes.max():.2f})")
            print(f"   - Longitudes: {len(longitudes)} valores ({longitudes.min():.2f} a {longitudes.max():.2f})")
            
            # Leer datos de precipitación
            print("🌧️  Leyendo datos de precipitación...")
            precipitation_data = f['Grid/precipitation'][:]
            print(f"   - Forma de datos de precipitación: {precipitation_data.shape}")
            
            # Crear meshgrid para coordenadas
            print("🗺️  Creando meshgrid de coordenadas...")
            lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
            
            # Aplanar los arrays para crear el DataFrame
            print("📊 Creando DataFrame...")
            lat_flat = lat_grid.flatten()
            lon_flat = lon_grid.flatten()
            precip_flat = precipitation_data.flatten()
            
            # Crear DataFrame
            df = pd.DataFrame({
                'latitude': lat_flat,
                'longitude': lon_flat,
                'precipitation': precip_flat
            })
            
            # Limpiar datos (remover valores inválidos)
            print("🧹 Limpiando datos...")
            initial_rows = len(df)
            
            # Remover NaN
            df = df.dropna()
            
            # Remover valores de precipitación inválidos (valores de relleno típicos en GPM)
            df = df[df['precipitation'] > -9999.0]  # Remover valores de relleno
            df = df[df['precipitation'] < 1000.0]   # Remover valores extremadamente altos
            
            print(f"   - Filas iniciales: {initial_rows:,}")
            print(f"   - Filas después de limpieza: {len(df):,}")
            
            # Estadísticas
            total_points = len(df)
            rain_points = len(df[df['precipitation'] > 0])
            max_precip = df['precipitation'].max()
            mean_precip = df[df['precipitation'] > 0]['precipitation'].mean() if rain_points > 0 else 0
            
            print(f"\n📈 Estadísticas:")
            print(f"   - Total de puntos válidos: {total_points:,}")
            print(f"   - Puntos con lluvia (>0): {rain_points:,} ({rain_points/total_points*100:.1f}%)")
            print(f"   - Precipitación máxima: {max_precip:.4f} mm/hr")
            print(f"   - Precipitación promedio (donde llueve): {mean_precip:.4f} mm/hr")
            
            # Guardar CSV
            print(f"💾 Guardando CSV: {output_file}")
            df.to_csv(output_file, index=False)
            
            # Mostrar muestra de datos
            print(f"\n📋 Muestra de datos (primeras 10 filas):")
            print(df.head(10).to_string(index=False))
            
            # Mostrar algunos puntos con lluvia si existen
            if rain_points > 0:
                rain_sample = df[df['precipitation'] > 0].head(5)
                print(f"\n🌧️  Muestra de puntos con lluvia:")
                print(rain_sample.to_string(index=False))
            
            print(f"\n✅ Conversión completada exitosamente!")
            return df
            
    except Exception as e:
        print(f"❌ Error procesando archivo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Procesar archivos descargados
if downloaded_files:
    print(f"\n🔄 Procesando {len(downloaded_files)} archivos descargados...")
    
    csv_files = []
    for i, file_path in enumerate(downloaded_files):
        if os.path.exists(file_path):
            # Crear nombre del archivo CSV de salida
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_csv = f"precipitation_data_{base_name}.csv"
            
            print(f"\n{'='*60}")
            print(f"ARCHIVO {i+1}/{len(downloaded_files)}")
            print(f"{'='*60}")
            
            df = convert_gpm_to_csv_with_coordinates(file_path, output_csv)
            if df is not None:
                csv_files.append(output_csv)
        else:
            print(f"⚠️  Archivo no encontrado: {file_path}")
    
    # Resumen final
    print(f"\n🎉 RESUMEN FINAL")
    print("=" * 50)
    print(f"📄 Archivos CSV generados: {len(csv_files)}")
    
    total_records = 0
    total_rain_points = 0
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df_summary = pd.read_csv(csv_file)
            rain_points = len(df_summary[df_summary['precipitation'] > 0])
            total_records += len(df_summary)
            total_rain_points += rain_points
            
            print(f"\n📊 {csv_file}:")
            print(f"   - Total registros: {len(df_summary):,}")
            print(f"   - Puntos con lluvia: {rain_points:,}")
            print(f"   - Precipitación máxima: {df_summary['precipitation'].max():.4f} mm/hr")
    
    print(f"\n📈 ESTADÍSTICAS GLOBALES:")
    print(f"   - Total de registros: {total_records:,}")
    print(f"   - Total de puntos con lluvia: {total_rain_points:,}")
    print(f"   - Porcentaje con lluvia: {total_rain_points/total_records*100:.2f}%")
    
    print(f"\n✅ ¡Archivos CSV listos para análisis!")
    print(f"📁 Ubicación: {os.getcwd()}")
    
else:
    print("❌ No hay archivos descargados para procesar")