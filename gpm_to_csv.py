import h5py
import pandas as pd
import numpy as np
import os
import glob

def gpm_to_csv(hdf5_file, output_file=None):
    """
    Convierte archivo GPM HDF5 a CSV con columnas: latitude, longitude, precipitation
    
    Args:
        hdf5_file (str): Ruta del archivo HDF5
        output_file (str): Nombre del archivo CSV de salida (opcional)
    
    Returns:
        str: Ruta del archivo CSV generado
    """
    with h5py.File(hdf5_file, 'r') as f:
        # Leer coordenadas y precipitación
        latitudes = np.array(f['Grid/lat'])
        longitudes = np.array(f['Grid/lon'])
        precipitation_data = np.array(f['Grid/precipitation'])
        
        # Si hay dimensión temporal, tomar el primer paso
        if len(precipitation_data.shape) == 3:
            precipitation_data = precipitation_data[0]
        
        # Crear meshgrid de coordenadas
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        
        # Crear DataFrame
        df = pd.DataFrame({
            'latitude': lat_grid.flatten(),
            'longitude': lon_grid.flatten(),
            'precipitation': precipitation_data.flatten()
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