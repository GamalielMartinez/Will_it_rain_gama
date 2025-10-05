
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
    short_name='GPM_3IMERGHHE',  # GPM IMERG Half Hourly
    cloud_hosted=True,
    temporal=("2025-10-01", "2025-10-05"),  # Solo 1 día para ejemplo
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
downloaded_files = earthaccess.download(granules, './data')
print("\nArchivos descargados:")
for file in downloaded_files:
    print(f"  - {file}")
