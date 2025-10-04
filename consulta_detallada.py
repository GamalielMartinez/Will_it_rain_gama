import earthaccess
import os
from dotenv import load_dotenv
import xarray as xr
import pandas as pd

load_dotenv()

# Configurar credenciales
os.environ['EARTHDATA_USERNAME'] = os.environ['EARTHDATA_USERNAME']
os.environ['EARTHDATA_PASSWORD'] = os.environ['EARTHDATA_PASSWORD']

print("Autenticando con NASA Earthdata...")
auth = earthaccess.login(strategy='environment')
print("✓ Autenticación exitosa!\n")

# =============================================================================
# CONSULTA POR REGIÓN ESPECÍFICA
# =============================================================================
print("\n" + "=" * 80)
print("BÚSQUEDA POR REGIÓN ESPECÍFICA (GOLFO DE MÉXICO)")
print("=" * 80)

# Coordenadas del Golfo de México
gulf_bbox = (-98.0, 18.0, -80.0, 31.0)

print(f"\nRegión: Golfo de México")
print(f"Coordenadas: {gulf_bbox}")
print(f"Buscando datos para enero 2024...")

granules_gulf = earthaccess.search_data(
    short_name='GPM_3IMERGHH',
    cloud_hosted=True,
    bounding_box=gulf_bbox,
    temporal=("2024-01-01", "2024-01-02"),
    count=5
)

print(f"\n✓ Se encontraron {len(granules_gulf)} archivos para esta región")

if len(granules_gulf) > 0:
    print("\nPrimeros 3 archivos:")
    for i, g in enumerate(granules_gulf[:3]):
        print(f"\n{i+1}. {g['umm']['GranuleUR']}")
        print(f"   Fecha: {g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']}")

# Descomenta estas líneas para descargar:
downloaded_files = earthaccess.download(granules_gulf[:2], './data')
print("\nArchivos descargados:")
for file in downloaded_files:
    print(f"  - {file}")

# Convertir HDF5 a CSV
print("\n" + "=" * 80)
print("CONVERSIÓN A CSV")
print("=" * 80)

import h5py

for file in downloaded_files:
    try:
        print(f"\nProcesando: {os.path.basename(file)}")
        
        # Abrir archivo HDF5
        with h5py.File(file, 'r') as hf:
            # Extraer precipitación (variable principal)
            precip_data = hf['Grid/precipitation'][:][0]  # Obtener primera dimensión temporal
            lat_data = hf['Grid/lat'][:]
            lon_data = hf['Grid/lon'][:]
            
            # Crear DataFrame con los datos
            # Aplanar los datos de precipitación y crear coordenadas correspondientes
            rows = []
            for i in range(len(lon_data)):
                for j in range(len(lat_data)):
                    precip_value = precip_data[i, j]
                    if precip_value > -9999:  # Filtrar valores no válidos
                        rows.append({
                            'lon': lon_data[i],
                            'lat': lat_data[j],
                            'precipitation_mm_hr': precip_value
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
