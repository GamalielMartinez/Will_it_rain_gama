import h5py
import numpy as np
import os

def quick_explore_hdf5(file_path):
    """
    Exploración rápida de archivo HDF5
    """
    print(f"📄 Archivo: {os.path.basename(file_path)}")
    print(f"📊 Tamaño: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
    print()
    
    with h5py.File(file_path, 'r') as f:
        
        print("🏗️  ESTRUCTURA:")
        print("-" * 40)
        
        def show_item(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 {name}/")
            elif isinstance(obj, h5py.Dataset):
                print(f"📄 {name}")
                print(f"   Forma: {obj.shape}")
                print(f"   Tipo: {obj.dtype}")
                
                # Mostrar valores para datos pequeños
                if obj.size <= 10:
                    try:
                        print(f"   Valores: {obj[:]}")
                    except:
                        pass
                elif len(obj.shape) == 1 and obj.size <= 100:
                    try:
                        data = obj[:]
                        print(f"   Rango: {data.min():.3f} a {data.max():.3f}")
                    except:
                        pass
        
        f.visititems(show_item)
        
        print(f"\n🌍 COORDENADAS:")
        print("-" * 40)
        if 'Grid/lat' in f:
            lat_data = f['Grid/lat'][:]
            print(f"Latitudes: {len(lat_data)} puntos ({lat_data.min():.2f}° a {lat_data.max():.2f}°)")
        
        if 'Grid/lon' in f:
            lon_data = f['Grid/lon'][:]
            print(f"Longitudes: {len(lon_data)} puntos ({lon_data.min():.2f}° a {lon_data.max():.2f}°)")
        
        print(f"\n🕒 TIEMPO:")
        print("-" * 40)
        if 'Grid/time' in f:
            time_data = f['Grid/time'][:]
            print(f"Timestampsz: {time_data}")
            
            # Convertir a fecha
            try:
                from datetime import datetime, timedelta
                gps_epoch = datetime(1980, 1, 6)
                readable_time = gps_epoch + timedelta(seconds=float(time_data[0]))
                print(f"Fecha: {readable_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            except:
                pass
        
        print(f"\n🌧️  PRECIPITACIÓN:")
        print("-" * 40)
        if 'Grid/precipitation' in f:
            precip = f['Grid/precipitation']
            print(f"Forma: {precip.shape}")
            
            # Analizar muestra
            try:
                sample = precip[:]
                if len(sample.shape) == 3:
                    sample = sample[0]  # Primer tiempo
                
                valid = sample[~np.isnan(sample)]
                rain_points = valid[valid > 0]
                
                print(f"Puntos válidos: {len(valid):,}")
                print(f"Puntos con lluvia: {len(rain_points):,} ({len(rain_points)/len(valid)*100:.1f}%)")
                if len(rain_points) > 0:
                    print(f"Precipitación max: {rain_points.max():.3f} mm/hr")
            except Exception as e:
                print(f"Error analizando: {str(e)}")
        
        print(f"\n📋 VARIABLES DISPONIBLES PARA CSV:")
        print("-" * 40)
        available_vars = []
        
        key_vars = ['Grid/lat', 'Grid/lon', 'Grid/precipitation', 'Grid/time',
                   'Grid/precipitationQualityIndex', 'Grid/randomError',
                   'Grid/probabilityLiquidPrecipitation']
        
        for var in key_vars:
            if var in f:
                available_vars.append(var)
                shape = f[var].shape
                print(f"✅ {var}: {shape}")
        
        # Variables de Intermediate
        intermediate_vars = []
        if 'Grid/Intermediate' in f:
            for item in f['Grid/Intermediate'].keys():
                var_path = f'Grid/Intermediate/{item}'
                if isinstance(f[var_path], h5py.Dataset):
                    intermediate_vars.append(var_path)
                    shape = f[var_path].shape
                    print(f"✅ {var_path}: {shape}")
        
        print(f"\n💡 RECOMENDACIONES:")
        print("-" * 40)
        print("Variables principales para CSV:")
        print("• Grid/lat, Grid/lon (coordenadas)")
        print("• Grid/precipitation (precipitación principal)")
        print("• Grid/time (tiempo)")
        
        if intermediate_vars:
            print("\nVariables adicionales disponibles:")
            for var in intermediate_vars:
                name = var.split('/')[-1]
                print(f"• {var} ({name})")

def main():
    # Buscar primer archivo HDF5
    import glob
    hdf5_files = glob.glob("./data/*.HDF5")
    
    if hdf5_files:
        quick_explore_hdf5(hdf5_files[0])
    else:
        print("❌ No se encontraron archivos HDF5 en ./data/")

if __name__ == "__main__":
    main()