import h5py
import numpy as np
import os
import glob

def explore_hdf5_simple(file_path):
    """
    Explorador simple de archivo HDF5
    """
    print(f"📄 ARCHIVO: {os.path.basename(file_path)}")
    print(f"📊 Tamaño: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
    print("=" * 60)
    
    with h5py.File(file_path, 'r') as f:
        
        print("🏗️  ESTRUCTURA COMPLETA:")
        print("-" * 40)
        
        def show_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📁 {name}/")
            elif isinstance(obj, h5py.Dataset):
                try:
                    print(f"📄 {name}")
                    print(f"   • Forma: {obj.shape}")
                    print(f"   • Tipo: {obj.dtype}")
                    
                    # Mostrar atributos si existen
                    if obj.attrs:
                        attrs = list(obj.attrs.keys())[:3]  # Primeros 3 atributos
                        if attrs:
                            print(f"   • Atributos: {', '.join(attrs)}")
                    
                    # Para datasets pequeños, mostrar info adicional
                    if hasattr(obj, 'size') and obj.size is not None and obj.size < 1000000:
                        try:
                            data = np.array(obj)
                            if np.issubdtype(data.dtype, np.number) and data.size > 1:
                                print(f"   • Rango: {data.min():.3f} a {data.max():.3f}")
                        except:
                            pass
                    
                except Exception as e:
                    print(f"📄 {name} (error leyendo: {str(e)[:30]})")
        
        f.visititems(show_structure)
        
        print(f"\n🌍 INFORMACIÓN DE COORDENADAS:")
        print("-" * 40)
        
        # Latitudes
        try:
            if 'Grid/lat' in f:
                lat_data = np.array(f['Grid/lat'])
                print(f"📍 Latitudes:")
                print(f"   • Cantidad: {len(lat_data)} puntos")
                print(f"   • Rango: {lat_data.min():.3f}° a {lat_data.max():.3f}°")
                if len(lat_data) > 1:
                    resolution = abs(lat_data[1] - lat_data[0])
                    print(f"   • Resolución: ~{resolution:.3f}°")
        except Exception as e:
            print(f"❌ Error leyendo latitudes: {str(e)}")
        
        # Longitudes
        try:
            if 'Grid/lon' in f:
                lon_data = np.array(f['Grid/lon'])
                print(f"📍 Longitudes:")
                print(f"   • Cantidad: {len(lon_data)} puntos")
                print(f"   • Rango: {lon_data.min():.3f}° a {lon_data.max():.3f}°")
                if len(lon_data) > 1:
                    resolution = abs(lon_data[1] - lon_data[0])
                    print(f"   • Resolución: ~{resolution:.3f}°")
        except Exception as e:
            print(f"❌ Error leyendo longitudes: {str(e)}")
        
        print(f"\n🕒 INFORMACIÓN TEMPORAL:")
        print("-" * 40)
        
        try:
            if 'Grid/time' in f:
                time_data = np.array(f['Grid/time'])
                print(f"⏰ Tiempo:")
                print(f"   • Cantidad: {len(time_data)} timestep(s)")
                print(f"   • Valores: {time_data}")
                
                # Intentar convertir a fecha
                try:
                    from datetime import datetime, timedelta
                    gps_epoch = datetime(1980, 1, 6)
                    readable_time = gps_epoch + timedelta(seconds=float(time_data[0]))
                    print(f"   • Fecha: {readable_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                except:
                    print(f"   • (No se pudo convertir a fecha)")
        except Exception as e:
            print(f"❌ Error leyendo tiempo: {str(e)}")
        
        print(f"\n🌧️  ANÁLISIS DE PRECIPITACIÓN:")
        print("-" * 40)
        
        precip_vars = ['Grid/precipitation', 'Grid/Intermediate/precipitation', 
                      'Grid/Intermediate/IRprecipitation', 'Grid/Intermediate/MWprecipitation']
        
        for var_name in precip_vars:
            try:
                if var_name in f:
                    precip_data = np.array(f[var_name])
                    print(f"💧 {var_name}:")
                    print(f"   • Forma: {precip_data.shape}")
                    
                    # Analizar datos (tomar muestra si es muy grande)
                    if len(precip_data.shape) == 3:
                        sample = precip_data[0]  # Primer timestep
                        print(f"   • Analizando primer timestep: {sample.shape}")
                    else:
                        sample = precip_data
                    
                    # Estadísticas básicas
                    valid_data = sample[~np.isnan(sample)]
                    rain_points = valid_data[valid_data > 0]
                    
                    print(f"   • Puntos válidos: {len(valid_data):,}")
                    print(f"   • Puntos con lluvia: {len(rain_points):,} ({len(rain_points)/len(valid_data)*100:.1f}%)")
                    
                    if len(rain_points) > 0:
                        print(f"   • Precipitación mín: {rain_points.min():.4f}")
                        print(f"   • Precipitación máx: {rain_points.max():.4f}")
                        print(f"   • Precipitación media: {rain_points.mean():.4f}")
                    
            except Exception as e:
                print(f"❌ Error analizando {var_name}: {str(e)}")
        
        print(f"\n📋 RESUMEN - VARIABLES PARA CSV:")
        print("-" * 40)
        
        available_for_csv = []
        
        # Variables principales
        main_vars = {
            'Grid/lat': 'Latitudes',
            'Grid/lon': 'Longitudes', 
            'Grid/time': 'Tiempo',
            'Grid/precipitation': 'Precipitación principal'
        }
        
        for var, desc in main_vars.items():
            if var in f:
                available_for_csv.append(var)
                shape = f[var].shape
                print(f"✅ {var} ({desc}): {shape}")
        
        # Variables adicionales interesantes
        additional_vars = {
            'Grid/precipitationQualityIndex': 'Índice de calidad',
            'Grid/randomError': 'Error aleatorio',
            'Grid/probabilityLiquidPrecipitation': 'Probabilidad lluvia líquida'
        }
        
        for var, desc in additional_vars.items():
            if var in f:
                available_for_csv.append(var)
                shape = f[var].shape
                print(f"✅ {var} ({desc}): {shape}")
        
        # Variables Intermediate
        if 'Grid/Intermediate' in f:
            print(f"\n🔬 Variables Intermediate disponibles:")
            try:
                intermediate_group = f['Grid/Intermediate']
                for item_name in intermediate_group.keys():
                    item_path = f'Grid/Intermediate/{item_name}'
                    if isinstance(intermediate_group[item_name], h5py.Dataset):
                        shape = intermediate_group[item_name].shape
                        print(f"   • {item_path}: {shape}")
            except Exception as e:
                print(f"   ❌ Error listando Intermediate: {str(e)}")
        
        print(f"\n💡 RECOMENDACIONES:")
        print("-" * 40)
        print("Para un CSV básico, usar:")
        print("1. Grid/lat + Grid/lon (coordenadas)")
        print("2. Grid/precipitation (datos principales)")
        print("3. Grid/time (información temporal)")
        print()
        print("Variables adicionales opcionales:")
        print("- Grid/precipitationQualityIndex (calidad)")
        print("- Grid/Intermediate/IRprecipitation (datos IR)")
        print("- Grid/Intermediate/MWprecipitation (datos microondas)")

def main():
    """Función principal"""
    print("🔍 EXPLORADOR SIMPLE DE ARCHIVOS HDF5")
    print("=" * 60)
    
    # Buscar archivos
    hdf5_files = glob.glob("./data/*.HDF5")
    
    if not hdf5_files:
        print("❌ No se encontraron archivos HDF5 en ./data/")
        return
    
    print(f"📁 Archivos encontrados: {len(hdf5_files)}")
    for i, file in enumerate(hdf5_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    print()
    
    # Analizar primer archivo
    explore_hdf5_simple(hdf5_files[0])
    
    if len(hdf5_files) > 1:
        print(f"\n📝 Nota: Se analizó solo el primer archivo.")
        print(f"   Los otros {len(hdf5_files)-1} archivos tienen estructura similar.")

if __name__ == "__main__":
    main()