import h5py
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

def gpm_to_csv(hdf5_file, output_file=None):
    """
    Convierte archivo GPM HDF5 a CSV con columnas: latitude, longitude, precipitation, time, datetime
    
    Args:
        hdf5_file (str): Ruta del archivo HDF5
        output_file (str): Nombre del archivo CSV de salida (opcional)
    
    Returns:
        str: Ruta del archivo CSV generado
    """
    with h5py.File(hdf5_file, 'r') as f:
        # Leer coordenadas, precipitación y tiempo
        latitudes = np.array(f['Grid/lat'])
        longitudes = np.array(f['Grid/lon'])
        precipitation_data = np.array(f['Grid/precipitation'])
        time_data = np.array(f['Grid/time'])
        
        # Si hay dimensión temporal, tomar el primer paso
        if len(precipitation_data.shape) == 3:
            precipitation_data = precipitation_data[0]
        
        # Crear meshgrid de coordenadas
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        # Preparar datos para DataFrame
        lat_flat = lat_grid.flatten()
        lon_flat = lon_grid.flatten()
        precip_flat = precipitation_data.flatten()
        
        # Procesar tiempo
        time_value = time_data[0] if len(time_data.shape) > 0 else time_data
        time_flat = np.full(len(lat_flat), time_value)
        
        # Convertir tiempo a formato legible
        # GPM usa diferentes epochs, intentar conversión
        try:
            # Epoch GPM desde 1980-01-06 00:00:00 (GPS time)
            gps_epoch = datetime(1980, 1, 6)
            readable_time = gps_epoch + timedelta(seconds=float(time_value))
            datetime_str = readable_time.strftime('%Y-%m-%d %H:%M:%S')
        except:
            try:
                # Epoch Unix estándar desde 1970-01-01
                readable_time = datetime.utcfromtimestamp(float(time_value))
                datetime_str = readable_time.strftime('%Y-%m-%d %H:%M:%S')
            except:
                # Si falla la conversión, usar el valor raw
                datetime_str = str(time_value)
        
        datetime_flat = np.full(len(lat_flat), datetime_str)
        
        # Crear DataFrame
        df = pd.DataFrame({
            'latitude': lat_flat,
            'longitude': lon_flat,
            'precipitation': precip_flat,
            'time': time_flat,
            'datetime': datetime_flat
        })
        
        # Remover valores inválidos
        df = df.dropna()
        df = df[df['precipitation'] >= 0]
        
        # Definir nombre del archivo de salida
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(hdf5_file))[0]
            output_file = f"{base_name}.csv"
        
        # Guardar CSV
        df.to_csv(output_file, index=False)
        
        return output_file

def convert_all_files():
    """Convierte todos los archivos HDF5 en la carpeta data/"""
    hdf5_files = glob.glob("./data/*.HDF5")
    
    for file_path in hdf5_files:
        csv_file = gpm_to_csv(file_path)
        print(f"Convertido: {csv_file}")

if __name__ == "__main__":
    convert_all_files()