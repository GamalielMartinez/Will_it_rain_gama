from nicegui import ui
import folium
from nicegui.elements.html import Html

@ui.page('/',title="BAM - Build Manager",response_timeout=15)
async def main_page():
    left_drawer()
    await header()
    await map_entry() 

async def header():
    with ui.header().classes('row flex h-20 items-center bg-blue-950') as head:
        ui.image('src/logo/nova_delta.png').classes('object-fit w-20 h-full')
        ui.label("Nova Delta").classes('text-2xl font-medium mb-2')

def left_drawer():
    with ui.left_drawer().classes('w-72') as drawer:
        ui.label("Parameters to show").classes('text-xl font-medium mb-2').style('font-weight: bold;')
        ui.switch('Temperature', value=True).classes('mb-2').props('left-label')
        ui.switch('Humidity', value=True).classes('mb-2').props('left-label')
        ui.switch('Pressure', value=True).classes('mb-2').props('left-label')
        ui.switch('Wind Speed', value=True).classes('mb-2').props('left-label')  
        ui.switch('Cloud Cover', value=True).classes('mb-2').props('left-label')  
        ui.switch('Precipitation', value=True).classes('mb-2').props('left-label')

async def map_entry():
    with ui.card().classes('m-4 p-4 w-full'):
        ui.label("Mapa Interactivo - Coordenadas en Consola").classes('text-xl font-bold mb-4')
        # Crear mapa con Folium
        def create_clickable_map():
            # Crear mapa centrado en el mundo
            m = folium.Map(
                location=[20, 0],  # Centro mundial
                zoom_start=2,
                tiles='OpenStreetMap'
            )
            
            # JavaScript mejorado para capturar clics
            click_script = """
            <script>
            function setupMapClickHandler() {
                // Buscar el mapa en el DOM
                setTimeout(function() {
                    // Buscar todas las instancias de mapas Leaflet
                    for (var key in window) {
                        if (key.indexOf('map_') === 0 && window[key] && window[key]._container) {
                            var map = window[key];
                            console.log('🗺️ Mapa encontrado y configurado para clics');
                            
                            // Agregar evento de clic
                            map.on('click', function(e) {
                                var lat = e.latlng.lat;
                                var lng = e.latlng.lng;
                                
                                // Imprimir en consola
                                console.log('🗺️ COORDENADAS CLICKEADAS:');
                                console.log('Latitud: ' + lat);
                                console.log('Longitud: ' + lng);
                                console.log('Coordenadas completas: [' + lat + ', ' + lng + ']');
                                console.log('─────────────────────────────────');
                                
                                // Crear popup
                                L.popup()
                                    .setLatLng(e.latlng)
                                    .setContent('📍 Coordenadas:<br><b>Lat:</b> ' + lat.toFixed(6) + '<br><b>Lng:</b> ' + lng.toFixed(6) + '<br><small>Ver consola para más detalles</small>')
                                    .openOn(map);
                            });
                            
                            return; // Salir cuando encontremos el mapa
                        }
                    }
                    
                    // Si no se encuentra, intentar de nuevo
                    console.log('� Buscando mapa, reintentando...');
                    setupMapClickHandler();
                }, 1000);
            }
            
            // Inicializar cuando el DOM esté listo
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', setupMapClickHandler);
            } else {
                setupMapClickHandler();
            }
            </script>
            """
            
            # Agregar el script al mapa
            m.get_root().html.add_child(folium.Element(click_script))
            
            return m
        
        # Crear y mostrar el mapa
        folium_map = create_clickable_map()
        map_html = folium_map._repr_html_()
        
        # Mostrar el mapa
        ui.html(map_html, sanitize=False).classes('w-full').style('height: 500px; border: 2px solid #ddd; border-radius: 8px;')
        
        # Instrucciones para el usuario
        ui.separator().classes('my-4')

        
       
        
        
if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title="Nova Delta Meteorology",
        favicon="src/logo/nova_delta.jpeg",
        port=8001,
        dark=True,
        reload=True,
        uvicorn_reload_excludes='.venv*',
        uvicorn_reload_dirs='src'
    )