import h5py
import numpy as np
import glob
import os
from datetime import datetime, timedelta

def explore_hdf5_detailed(file_path):
    """
    Explora completamente un archivo HDF5 y muestra toda la información disponible
    """
    print(f"🔍 ANÁLISIS COMPLETO DEL ARCHIVO HDF5")
    print("=" * 80)
    print(f"📄 Archivo: {os.path.basename(file_path)}")
    print(f"📁 Ruta: {file_path}")
    print(f"📊 Tamaño: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print()

    with h5py.File(file_path, 'r') as f:
        
        # 1. Estructura general
        print("🏗️  ESTRUCTURA GENERAL")
        print("-" * 50)
        
        def print_structure(name, obj, level=0):
            indent = "  " * level
            if isinstance(obj, h5py.Group):
                print(f"{indent}📁 {name}/")
                # Mostrar atributos del grupo
                if obj.attrs:
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"{indent}   📌 {attr_name}: {attr_value}")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📄 {name}")
                print(f"{indent}   📊 Forma: {obj.shape}")
                print(f"{indent}   🔢 Tipo: {obj.dtype}")
                print(f"{indent}   💾 Tamaño: {obj.size} elementos")
                
                # Mostrar atributos del dataset
                if obj.attrs:
                    print(f"{indent}   📌 Atributos:")
                    for attr_name, attr_value in obj.attrs.items():
                        if isinstance(attr_value, (bytes, np.bytes_)):
                            attr_value = attr_value.decode('utf-8', errors='ignore')
                        print(f"{indent}      • {attr_name}: {attr_value}")
                
                # Mostrar rango de valores para datasets pequeños
                if obj.size < 100000 and len(obj.shape) <= 2:
                    try:
                        data = obj[:]
                        if np.issubdtype(obj.dtype, np.number):
                            print(f"{indent}   📈 Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
                            if hasattr(data, 'shape') and len(data.shape) > 0:
                                non_zero = data[data != 0] if np.any(data != 0) else []
                                if len(non_zero) > 0:
                                    print(f"{indent}   🎯 Valores no-cero: {len(non_zero)} ({len(non_zero)/data.size*100:.1f}%)")
                    except:
                        pass
        
        def visit_func(name, obj):
            level = name.count('/')
            print_structure(name, obj, level)
        
        f.visititems(visit_func)
        
        # 2. Información específica de coordenadas
        print(f"\n🌍 INFORMACIÓN DE COORDENADAS")
        print("-" * 50)
        
        coord_info = {}
        for coord_name in ['Grid/lat', 'Grid/lon', 'Grid/latv', 'Grid/lonv']:
            if coord_name in f:
                data = f[coord_name][:]
                coord_info[coord_name] = data
                print(f"📍 {coord_name}:")
                print(f"   • Valores: {len(data)} puntos")
                print(f"   • Rango: {data.min():.4f} a {data.max():.4f}")
                if len(data) > 1:
                    resolution = abs(data[1] - data[0])
                    print(f"   • Resolución: ~{resolution:.4f}°")
        
        # 3. Información temporal
        print(f"\n🕒 INFORMACIÓN TEMPORAL")
        print("-" * 50)
        
        if 'Grid/time' in f:
            time_data = f['Grid/time'][:]
            print(f"⏰ Grid/time:")
            print(f"   • Valores: {len(time_data)}")
            print(f"   • Timestamp raw: {time_data}")
            
            # Intentar convertir a fecha legible
            for i, time_val in enumerate(time_data):
                print(f"   • Tiempo {i+1}:")
                
                # Método 1: GPS epoch (1980-01-06)
                try:
                    gps_epoch = datetime(1980, 1, 6)
                    readable_time = gps_epoch + timedelta(seconds=float(time_val))
                    print(f"     - GPS epoch: {readable_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                except:
                    pass
                
                # Método 2: Unix epoch (1970-01-01)
                try:
                    readable_time = datetime.utcfromtimestamp(float(time_val))
                    print(f"     - Unix epoch: {readable_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                except:
                    pass
        
        if 'Grid/time_bnds' in f:
            time_bounds = f['Grid/time_bnds'][:]
            print(f"⏰ Grid/time_bnds:")
            print(f"   • Forma: {time_bounds.shape}")
            print(f"   • Límites: {time_bounds}")
        
        # 4. Variables de precipitación
        print(f"\n🌧️  VARIABLES DE PRECIPITACIÓN")
        print("-" * 50)
        
        precip_vars = []
        for name in f.keys():
            def find_precip(obj_name, obj):
                if isinstance(obj, h5py.Dataset) and 'precip' in obj_name.lower():
                    precip_vars.append(obj_name)
            
            f[name].visititems(find_precip)
        
        for var_name in precip_vars:
            if var_name in f:
                data = f[var_name]
                print(f"💧 {var_name}:")
                print(f"   • Forma: {data.shape}")
                print(f"   • Tipo: {data.dtype}")
                
                # Analizar datos de precipitación
                try:
                    sample_data = data[:]
                    if len(sample_data.shape) == 3:
                        sample_data = sample_data[0]  # Tomar primer tiempo
                    
                    # Estadísticas
                    valid_data = sample_data[~np.isnan(sample_data)]
                    positive_data = valid_data[valid_data > 0]
                    
                    print(f"   • Puntos válidos: {len(valid_data):,}")
                    print(f"   • Puntos con lluvia: {len(positive_data):,} ({len(positive_data)/len(valid_data)*100:.1f}%)")
                    
                    if len(positive_data) > 0:
                        print(f"   • Precipitación mín: {positive_data.min():.4f}")
                        print(f"   • Precipitación máx: {positive_data.max():.4f}")
                        print(f"   • Precipitación media: {positive_data.mean():.4f}")
                except Exception as e:
                    print(f"   • Error analizando datos: {str(e)}")
        
        # 5. Otras variables interesantes
        print(f"\n🔬 OTRAS VARIABLES DE INTERÉS")
        print("-" * 50)
        
        interesting_vars = []
        for name in ['Quality', 'Error', 'Probability', 'Influence', 'Source']:
            def find_interesting(obj_name, obj):
                if isinstance(obj, h5py.Dataset) and name.lower() in obj_name.lower():
                    interesting_vars.append(obj_name)
            
            f.visititems(find_interesting)
        
        for var_name in set(interesting_vars):
            if var_name in f:
                data = f[var_name]
                print(f"🔍 {var_name}:")
                print(f"   • Forma: {data.shape}")
                print(f"   • Tipo: {data.dtype}")
                
                # Mostrar algunos valores únicos para variables categóricas
                try:
                    sample = data[:]
                    if sample.size < 10000:  # Solo para datasets pequeños
                        unique_vals = np.unique(sample)
                        if len(unique_vals) < 20:
                            print(f"   • Valores únicos: {unique_vals}")
                except:
                    pass
        
        # 6. Atributos globales del archivo
        print(f"\n📋 ATRIBUTOS GLOBALES DEL ARCHIVO")
        print("-" * 50)
        
        if f.attrs:
            for attr_name, attr_value in f.attrs.items():
                if isinstance(attr_value, (bytes, np.bytes_)):
                    attr_value = attr_value.decode('utf-8', errors='ignore')
                print(f"🏷️  {attr_name}: {attr_value}")
        else:
            print("No hay atributos globales")
        
        # 7. Resumen para CSV
        print(f"\n📊 RESUMEN PARA CONVERSIÓN A CSV")
        print("-" * 50)
        print("Variables disponibles para extraer:")
        
        if 'Grid/lat' in f and 'Grid/lon' in f:
            lat_count = len(f['Grid/lat'])
            lon_count = len(f['Grid/lon'])
            total_points = lat_count * lon_count
            print(f"✅ Coordenadas: {lat_count} latitudes × {lon_count} longitudes = {total_points:,} puntos")
        
        if 'Grid/time' in f:
            time_count = len(f['Grid/time'])
            print(f"✅ Tiempo: {time_count} paso(s) temporal(es)")
        
        if precip_vars:
            print(f"✅ Variables de precipitación: {len(precip_vars)}")
            for var in precip_vars:
                print(f"   • {var}")
        
        if interesting_vars:
            print(f"✅ Variables adicionales: {len(set(interesting_vars))}")
            for var in set(interesting_vars):
                print(f"   • {var}")

def main():
    """
    Función principal para explorar archivos HDF5
    """
    print("🔍 EXPLORADOR DE ARCHIVOS HDF5 - DATOS GPM")
    print("=" * 80)
    
    # Buscar archivos HDF5
    hdf5_files = glob.glob("./data/*.HDF5")
    
    if not hdf5_files:
        print("❌ No se encontraron archivos HDF5 en ./data/")
        return
    
    print(f"📁 Archivos encontrados: {len(hdf5_files)}")
    for i, file in enumerate(hdf5_files, 1):
        print(f"   {i}. {os.path.basename(file)}")
    
    # Analizar primer archivo (o todos si quieres)
    for i, file_path in enumerate(hdf5_files[:1]):  # Solo el primero
        print(f"\n{'='*80}")
        print(f"ANALIZANDO ARCHIVO {i+1}/{len(hdf5_files)}")
        print(f"{'='*80}")
        explore_hdf5_detailed(file_path)
    
    print(f"\n✅ Análisis completado!")
    print(f"💡 Usa esta información para decidir qué variables extraer en tu CSV")

if __name__ == "__main__":
    main()