
import earthaccess
import os
from dotenv import load_dotenv

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

# Convertir HDF5 a CSV
print("\n" + "=" * 70)
print("CONVERSIÓN A CSV")
print("=" * 70)

import h5py
import pandas as pd

for file in downloaded_files:
    try:
        print(f"\nProcesando: {os.path.basename(file)}")
        
        # Abrir archivo HDF5
        with h5py.File(file, 'r') as hf:
            # Extraer precipitación (variable principal)
            precip_data = hf['Grid/precipitation'][:]
            lat_data = hf['Grid/lat'][:]
            lon_data = hf['Grid/lon'][:]
            
            # Crear DataFrame con los datos
            # Aplanar los datos de precipitación y crear coordenadas correspondientes
            rows = []
            for i in range(precip_data.shape[0]):
                for j in range(precip_data.shape[1]):
                    if precip_data[i, j] >= 0:  # Filtrar valores negativos (no data)
                        rows.append({
                            'lat': lat_data[i],
                            'lon': lon_data[j],
                            'precipitation_mm_hr': precip_data[i, j]
                        })
            
            df = pd.DataFrame(rows)
            
            # Guardar como CSV
            csv_filename = file.replace('.HDF5', '.csv').replace('.nc4', '.csv')
            df.to_csv(csv_filename, index=False)
            print(f"  ✓ Convertido a CSV: {os.path.basename(csv_filename)}")
            print(f"  ✓ Registros: {len(df)}")
            
            # Opcional: Eliminar archivo HDF5 original
            os.remove(file)
            print(f"  ✓ Archivo HDF5 eliminado")
        
    except Exception as e:
        print(f"  ✗ Error al procesar {file}: {e}")

print("\n✓ Script completado exitosamente!")