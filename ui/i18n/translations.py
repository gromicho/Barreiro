"""Translation dictionaries for the UI (strings may include Unicode accents)."""

SUPPORTED_LANGS: dict[str, str] = {
    'nl': 'Nederlands',
    'en': 'English',
    'pt': 'Português',
}

TRANSLATIONS: dict[str, dict[str, str]] = {
    'en': {
        'addresses_label': 'Addresses (one per line):',
        'app_title': 'Visit route optimization\nprivate use only by{name}\nin {city} region',
        'cache_cleared_ok': 'Geocoding cache cleared.',
        'clear_cache': 'Clear geocoding cache (Dropbox)',
        'clear_failed': 'Clear failed: {error}',
        'diagnostics_title': 'Geocoding and snapping (diagnostics)',
        'dist_matrix': 'Computing distance matrix...',
        'dist_to_node_line': '- Distance to node: `{dist_m}` meters',
        'distance_matrix_title': 'Road network distance matrix (km)',
        'duplicate_coords_warning': 'Warning: multiple addresses were geocoded to exactly the same '
                                    'coordinate. This may indicate ambiguous input.',
        'estimated_total_km': 'Estimated total length (km): {km}',
        'geocode_error': 'Error while geocoding: {error}',
        'geocode_line': '- Geocode: lat `{lat}`, lon `{lon}`',
        'geocode_unexpected': 'Unexpected error while geocoding: {error}',
        'geocoding': 'Geocoding addresses...',
        'gurobi': 'Solving route optimally with Gurobi...',
        'input_address': 'Input address:',
        'instructions': 'Enter one address per line within the Capelle region.\n'
                        '\n'
                        '- The first address is the start location.\n'
                        '- For an open route, the last address is the end location.\n'
                        '\n'
                        'The app:\n'
                        '1. Geocodes addresses (Google)\n'
                        '2. Snaps to a small road network\n'
                        '3. Computes a distance matrix (km)\n'
                        '4. Solves exactly with Gurobi\n'
                        '5. Builds a Google Maps link.\n',
        'language_label': 'Language',
        'loading_network': 'Loading road network...',
        'maps_compare_title': 'Routes on map (comparison)',
        'maps_url_error': 'Error building the Google Maps URL: {error}',
        'nearest_node_line': '- Nearest network node: `{node_id}`',
        'need_three_open': 'For an open route, at least three addresses are required (start, '
                           'intermediate, end).',
        'need_two': 'Please provide at least two addresses.',
        'network_load_error': 'Error while loading the network: {error}',
        'off': 'Off',
        'on': 'On',
        'open_in_maps': 'Open in Google Maps',
        'open_nav': 'Open in navigation app',
        'opt_order': 'Optimized order',
        'optimize': 'Optimize route',
        'order_title': 'Optimized visit order (road network)',
        'orig_order': 'Original order',
        'reload_addresses': 'Reload addresses (Dropbox)',
        'reload_failed': 'Reload failed: {error}',
        'reloaded_ok': 'Reloaded.',
        'road_overlay': 'Draw route on road network',
        'route_closed': 'Closed tour (start and end at the first address)',
        'route_open': 'Open route (start at the first address, end at the last address)',
        'route_type': 'Route type',
        'save_addresses': 'Save addresses (Dropbox)',
        'save_failed': 'Save failed: {error}',
        'saved_ok': 'Saved to Dropbox.',
        'snap_error': 'Error while snapping to the network: {error}',
        'snapping': 'Snapping addresses to the road network...',
        'timinglog_caption': 'A full history is also written to routing_time_log.txt in the app '
                             'folder.',
        'timinglog_expander': 'Show detailed timing',
        'timinglog_title': 'Timing log for this run',
        'too_far_error': 'At least one address is too far from the available road network (more '
                         'than {km} km).',
        'total_distance_km': 'Total distance (km):',
        'ui_full': 'Full',
        'ui_mode': 'Interface mode',
        'ui_simple': 'Simple',
        'unreachable_error': 'Unreachable locations detected: {error}',
        'view_in_maps': 'View point in Google Maps',
    },
    'nl': {
        'addresses_label': 'Adressen (een per regel):',
        'app_title': 'Bezoekroute-optimalisatie\nexclusief voor {name}\neigen gebruik\nin regio {city}',
        'cache_cleared_ok': 'Geocoding cache gewist.',
        'clear_cache': 'Wis geocoding cache (Dropbox)',
        'clear_failed': 'Wissen mislukt: {error}',
        'diagnostics_title': 'Geocoding en snapping (diagnostiek)',
        'dist_matrix': 'Afstandsmatrix berekenen...',
        'dist_to_node_line': '- Afstand tot node: `{dist_m}` meter',
        'distance_matrix_title': 'Afstandsmatrix in het wegennet (km)',
        'duplicate_coords_warning': 'Let op: meerdere adressen zijn naar exact dezelfde coordinaat '
                                    'gegeocoderd. Dit kan duiden op onduidelijke invoer.',
        'estimated_total_km': 'Geschatte totale lengte (km): {km}',
        'geocode_error': 'Fout tijdens het geocoderen: {error}',
        'geocode_line': '- Geocode: lat `{lat}`, lon `{lon}`',
        'geocode_unexpected': 'Onverwachte fout tijdens geocoderen: {error}',
        'geocoding': 'Adressen geocoderen...',
        'gurobi': 'Route optimaal oplossen met Gurobi...',
        'input_address': 'Invoeradres:',
        'instructions': 'Voer een adres per regel in binnen de regio Capelle.\n'
                        '\n'
                        '- Het eerste adres is de startlocatie.\n'
                        '- Bij een open traject is het laatste adres de eindlocatie.\n'
                        '\n'
                        'De app:\n'
                        '1. Geocodeert adressen (Google)\n'
                        '2. Snapt naar een mini-wegennet\n'
                        '3. Berekent een afstandsmatrix (km)\n'
                        '4. Lost exact op met Gurobi\n'
                        '5. Genereert een Google Maps link.\n',
        'language_label': 'Taal',
        'loading_network': 'Wegennetwerk laden...',
        'maps_compare_title': 'Routes op kaart (vergelijking)',
        'maps_url_error': 'Fout bij het opbouwen van de Google Maps URL: {error}',
        'nearest_node_line': '- Dichtstbijzijnde netwerk-node: `{node_id}`',
        'need_three_open': 'Voor een open traject zijn minstens drie adressen nodig (start, '
                           'tussenadres, einde).',
        'need_two': 'Geef minstens twee adressen op.',
        'network_load_error': 'Fout tijdens het laden van het netwerk: {error}',
        'off': 'Uit',
        'on': 'Aan',
        'open_in_maps': 'Open in Google Maps',
        'open_nav': 'Open in navigatie-app',
        'opt_order': 'Geoptimaliseerde volgorde',
        'optimize': 'Optimaliseer route',
        'order_title': 'Geoptimaliseerde bezoekvolgorde (wegennet)',
        'orig_order': 'Oorspronkelijke volgorde',
        'reload_addresses': 'Herlaad adressen (Dropbox)',
        'reload_failed': 'Herladen mislukt: {error}',
        'reloaded_ok': 'Opnieuw geladen.',
        'road_overlay': 'Route over het wegennet tekenen',
        'route_closed': 'Gesloten rondrit (start en einde bij het eerste adres)',
        'route_open': 'Open traject (start bij het eerste adres, einde bij het laatste adres)',
        'route_type': 'Routetype',
        'save_addresses': 'Bewaar adressen (Dropbox)',
        'save_failed': 'Opslaan mislukt: {error}',
        'saved_ok': 'Opgeslagen in Dropbox.',
        'snap_error': 'Fout tijdens het koppelen aan het wegennet: {error}',
        'snapping': 'Adressen koppelen aan het wegennet...',
        'timinglog_caption': 'De volledige geschiedenis wordt ook weggeschreven naar '
                             'routing_time_log.txt in de app-map.',
        'timinglog_expander': 'Toon gedetailleerde timing',
        'timinglog_title': 'Timinglog voor deze run',
        'too_far_error': 'Minstens een adres ligt te ver van het beschikbare wegennet (meer dan '
                         '{km} km).',
        'total_distance_km': 'Totale afstand (km):',
        'ui_full': 'Volledig',
        'ui_mode': 'Interfacemodus',
        'ui_simple': 'Eenvoudig',
        'unreachable_error': 'Niet-bereikbare locaties gedetecteerd: {error}',
        'view_in_maps': 'Bekijk punt in Google Maps',
    },
    'pt': {
        'addresses_label': 'Endereços (um por linha):',
        'app_title': 'Otimização de rotas de visitas\nuso privado {name}\nregião de {city}',
        'cache_cleared_ok': 'Cache de geocodificação limpa.',
        'clear_cache': 'Limpar cache de geocodificação (Dropbox)',
        'clear_failed': 'Falha ao limpar: {error}',
        'diagnostics_title': 'Geocodificação e ajuste à rede (diagnóstico)',
        'dist_matrix': 'A calcular a matriz de distâncias...',
        'dist_to_node_line': '- Distância ao nó: `{dist_m}` metros',
        'distance_matrix_title': 'Matriz de distâncias na rede viária (km)',
        'duplicate_coords_warning': 'Aviso: vários endereços foram geocodificados para a mesma '
                                    'coordenada. Isto pode indicar uma entrada ambígua.',
        'estimated_total_km': 'Comprimento total estimado (km): {km}',
        'geocode_error': 'Erro ao geocodificar: {error}',
        'geocode_line': '- Geocódigo: lat `{lat}`, lon `{lon}`',
        'geocode_unexpected': 'Erro inesperado ao geocodificar: {error}',
        'geocoding': 'A geocodificar endereços...',
        'gurobi': 'A resolver a rota com o Gurobi...',
        'input_address': 'Endereço:',
        'instructions': 'Introduza um endereço por linha na região de Capelle.\n'
                        '\n'
                        '- O primeiro endereço é o ponto de partida.\n'
                        '- Numa rota aberta, o último endereço é o destino.\n'
                        '\n'
                        'A aplicação:\n'
                        '1. Geocodifica endereços (Google)\n'
                        '2. Ajusta a uma pequena rede viária\n'
                        '3. Calcula uma matriz de distâncias (km)\n'
                        '4. Resolve exatamente com Gurobi\n'
                        '5. Gera uma ligação do Google Maps.\n',
        'language_label': 'Idioma',
        'loading_network': 'A carregar a rede viária...',
        'maps_compare_title': 'Rotas no mapa (comparação)',
        'maps_url_error': 'Erro ao construir o URL do Google Maps: {error}',
        'nearest_node_line': '- Nó mais próximo na rede: `{node_id}`',
        'need_three_open': 'Para uma rota aberta são necessários pelo menos três endereços '
                           '(início, intermédio, fim).',
        'need_two': 'Indique pelo menos dois endereços.',
        'network_load_error': 'Erro ao carregar a rede: {error}',
        'off': 'Desligado',
        'on': 'Ligado',
        'open_in_maps': 'Abrir no Google Maps',
        'open_nav': 'Abrir na aplicação de navegação',
        'opt_order': 'Ordem otimizada',
        'optimize': 'Otimizar rota',
        'order_title': 'Ordem otimizada de visita (rede viária)',
        'orig_order': 'Ordem original',
        'reload_addresses': 'Recarregar endereços (Dropbox)',
        'reload_failed': 'Falha ao recarregar: {error}',
        'reloaded_ok': 'Recarregado.',
        'road_overlay': 'Desenhar rota na rede viária',
        'route_closed': 'Circuito fechado (início e fim no primeiro endereço)',
        'route_open': 'Rota aberta (início no primeiro endereço, fim no último endereço)',
        'route_type': 'Tipo de rota',
        'save_addresses': 'Guardar endereços (Dropbox)',
        'save_failed': 'Falha ao guardar: {error}',
        'saved_ok': 'Guardado no Dropbox.',
        'snap_error': 'Erro ao associar à rede: {error}',
        'snapping': 'A associar endereços à rede viária...',
        'timinglog_caption': 'Também é guardado um histórico completo em routing_time_log.txt na '
                             'pasta da aplicação.',
        'timinglog_expander': 'Mostrar tempos detalhados',
        'timinglog_title': 'Registo de tempos desta execução',
        'too_far_error': 'Pelo menos um endereço está demasiado longe da rede viária disponível '
                         '(mais de {km} km).',
        'total_distance_km': 'Distância total (km):',
        'ui_full': 'Completo',
        'ui_mode': 'Modo de interface',
        'ui_simple': 'Simples',
        'unreachable_error': 'Locais não alcançáveis detetados: {error}',
        'view_in_maps': 'Ver ponto no Google Maps',
    },
}
