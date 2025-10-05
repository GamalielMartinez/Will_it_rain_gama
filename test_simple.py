from gpm_to_csv import gpm_to_csv
import os

# Probar con un archivo específico
hdf5_file = "./data/3B-HHR.MS.MRG.3IMERG.20240101-S000000-E002959.0000.V07B.HDF5"

if os.path.exists(hdf5_file):
    print(f"Convirtiendo: {hdf5_file}")
    csv_file = gpm_to_csv(hdf5_file, "test_simple.csv")
    print(f"CSV generado: {csv_file}")
    
    # Verificar el CSV
    import pandas as pd
    df = pd.read_csv(csv_file)
    print(f"Filas: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print("Primeras 5 filas:")
    print(df.head())
else:
    print("Archivo HDF5 no encontrado")