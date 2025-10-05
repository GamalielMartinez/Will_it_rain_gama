import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Carga las variables de entorno desde el archivo .env

# --- Configuración (Modifica estos valores) ---
# Nombre del archivo que contiene las URLs, una por línea
URL_FILE = 'urls.txt'
# Carpeta donde se guardarán los archivos descargados
OUTPUT_DIR = 'D:/DatosDescargados'
# El token de autorización opcional (si no se usa, déjalo como None)
AUTH_TOKEN = os.environ['EARTHDATA_TOKEN']  # Reemplaza 'abc' con tu token real o deja None

# Configuraciones de la sesión
SESSION = requests.Session()
COOKIES_FILE = 'D:/.urs_cookies' # Archivo de cookies, si es necesario


def load_urls(file_path):
    """Lee y devuelve una lista de URLs desde un archivo."""
    try:
        with open(file_path, 'r') as f:
            # Filtra líneas vacías o de solo espacios en blanco
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: El archivo de URLs '{file_path}' no fue encontrado.")
        return []

def get_filename_from_response(response):
    """Intenta obtener el nombre del archivo del encabezado Content-Disposition o de la URL."""
    # 1. Intentar obtener el nombre del encabezado Content-Disposition (similar a curl -J)
    cd = response.headers.get('content-disposition')
    if cd:
        # Esto es una manera simple de parsear el nombre de archivo del encabezado
        try:
            return cd.split('filename=')[1].strip('"')
        except IndexError:
            pass # Si falla el parseo, sigue al siguiente método

    # 2. Usar el nombre de archivo de la URL (similar a curl -O)
    from urllib.parse import urlparse
    parsed_url = urlparse(response.url)
    return os.path.basename(parsed_url.path)


def download_file(url, output_path, headers=None, session=None):
    """Descarga un archivo desde una URL usando una sesión."""
    session = session or requests.Session()
    
    print(f"\n-> Intentando descargar: {url}")
    
    try:
        # Usa stream=True para descargar archivos grandes eficientemente
        response = session.get(url, headers=headers, stream=True, allow_redirects=True) # Similar a curl -L
        response.raise_for_status() # Lanza una excepción para códigos de error HTTP (4xx o 5xx)

        # Determinar el nombre del archivo
        filename = get_filename_from_response(response)
        full_path = os.path.join(output_path, filename)

        # Escribir el contenido en el archivo
        with open(full_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: # Filtrar fragmentos keep-alive
                    f.write(chunk)

        print(f"✅ Éxito: Archivo guardado como '{full_path}'")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"❌ Error HTTP para {url}: {e}")
        print("   Puede que necesites configurar tu archivo .netrc o la autenticación.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado al descargar {url}: {e}")
        
    return False


def main():
    # Asegúrate de que el directorio de salida exista
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    urls = load_urls(URL_FILE)
    if not urls:
        print("No hay URLs para procesar. Abortando.")
        return

    # 1. Configurar los headers de autorización si hay token
    custom_headers = {}
    if AUTH_TOKEN:
        custom_headers['Authorization'] = f'bearer {AUTH_TOKEN}'
        print("Configurando autorización con token...")
        
    # Nota sobre credenciales URS (similar a curl -n):
    # La librería 'requests' no tiene un equivalente directo y fácil a curl -n (lectura de .netrc).
    # La autenticación URS a menudo requiere un flujo de inicio de sesión que guarda cookies.
    # Para este script, confiaremos en que el servidor te pide credenciales o que uses el token.
    print("Iniciando descargas...")

    total_success = 0
    
    for url in urls:
        if download_file(url, OUTPUT_DIR, headers=custom_headers, session=SESSION):
            total_success += 1
            
    print(f"\n--- Resumen ---")
    print(f"Archivos intentados: {len(urls)}")
    print(f"Descargas exitosas: {total_success}")


if __name__ == "__main__":
    main()