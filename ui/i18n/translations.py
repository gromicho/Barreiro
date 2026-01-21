"""
Translation dictionaries for the UI (strings may include Unicode accents).
"""

SUPPORTED_LANGS: dict[str, str] = {
    'nl': 'Nederlands',
    'en': 'English',
    'pt': 'Português',
}

TRANSLATIONS: dict[str, dict[str, str]] = {
    'en': {
        # --- existing keys (unchanged) ---
        'addresses_label': 'Addresses (one per line):',
        'app_title': 'Visit route optimization\nprivate use only by {name}\nin {city} region',
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
        'version_label': 'Saved address versions',
        'load_version': 'Load version',
        'loaded_version_ok': 'Loaded version {version}.',
        'load_version_failed': 'Load failed: {error}',
        'no_versions': 'No saved versions found.',

        # --- widgets.py / camera OCR additions ---
        'camera_ocr_title': 'Camera OCR',
        'camera_ocr_take_photo': 'Take a photo of the address note',
        'ocr_extracting': 'Extracting addresses...',
        'ocr_failed': 'OCR failed: {error}',
        'ocr_none_found': 'No addresses found.',
        'ocr_debug_title': 'OCR debug',
        'debug_saved_photo': 'Saved debug photo: {path}',
        'debug_dropbox_path': 'Dropbox path: {path}',
        'dropbox_photo_save_failed': 'Dropbox photo save failed (continuing): {error}',
        'ocr_loaded_addresses_ok': 'Loaded {count} addresses into the input box.',
        'apply': 'Apply',
        'cancel': 'Cancel',
    },

    'nl': {
        # --- existing keys (unchanged) ---
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
        'version_label': 'Opgeslagen versies',
        'load_version': 'Laad versie',
        'loaded_version_ok': 'Versie {version} geladen.',
        'load_version_failed': 'Laden mislukt: {error}',
        'no_versions': 'Geen opgeslagen versies gevonden.',

        # --- widgets.py / camera OCR additions ---
        'camera_ocr_title': 'Camera OCR',
        'camera_ocr_take_photo': 'Maak een foto van het adressenbriefje',
        'ocr_extracting': 'Adressen extraheren...',
        'ocr_failed': 'OCR mislukt: {error}',
        'ocr_none_found': 'Geen adressen gevonden.',
        'ocr_debug_title': 'OCR debug',
        'debug_saved_photo': 'Debugfoto opgeslagen: {path}',
        'debug_dropbox_path': 'Dropbox-pad: {path}',
        'dropbox_photo_save_failed': 'Opslaan van foto in Dropbox mislukt (gaat door): {error}',
        'ocr_loaded_addresses_ok': '{count} adressen in het invoerveld geladen.',
        'apply': 'Toepassen',
        'cancel': 'Annuleren',
    },    

    'pt': {
        # --- existing keys (Portuguese corrected with accents) ---
        'addresses_label': 'Endereços (um por linha):',
        'app_title': 'Otimização de rotas de visitas\nuso privado de {name}\nna região de {city}',
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

        # Corrected accents:
        'version_label': 'Versões guardadas',
        'load_version': 'Carregar versão',
        'loaded_version_ok': 'Versão {version} carregada.',
        'load_version_failed': 'Falha ao carregar: {error}',
        'no_versions': 'Não existem versões guardadas.',

        # --- widgets.py / camera OCR additions (Portuguese with accents) ---
        'camera_ocr_title': 'OCR por câmara',
        'camera_ocr_take_photo': 'Tire uma foto da nota com os endereços',
        'ocr_extracting': 'A extrair endereços...',
        'ocr_failed': 'Falha no OCR: {error}',
        'ocr_none_found': 'Não foram encontrados endereços.',
        'ocr_debug_title': 'Depuração do OCR',
        'debug_saved_photo': 'Foto de depuração guardada: {path}',
        'debug_dropbox_path': 'Caminho no Dropbox: {path}',
        'dropbox_photo_save_failed': 'Falha ao guardar a foto no Dropbox (a continuar): {error}',
        'ocr_loaded_addresses_ok': 'Foram carregados {count} endereços na caixa de texto.',
        'apply': 'Aplicar',
        'cancel': 'Cancelar',
    },
}

# Add these keys to each language inside TRANSLATIONS (keeping your existing ones).
# I’m showing just the NEW/ALIASED entries to paste into each dict.

TRANSLATIONS['en'].update(
    {
        # --- app.py additions ---
        'input_preview_caption': 'Input preview (after home + de-dup)',
        'col_num': '#',
        'col_address': 'Address',
        'col_lat': 'Lat',
        'col_lon': 'Lon',
        'col_snap_dist_km': 'Snap dist (km)',
        'max_snap_distance_caption': 'Max snap distance: {km} km',
        'geocoded_title': 'Geocoding and snapping',
        'snapping_overview_title': 'Snapping overview',
        'snap_dist_km_col': 'Snap dist (km)',
        'order_title_simple': 'Optimized order',
        'dist_matrix_expander': 'Distance matrix',
        'maps_plots_expander': 'Maps / plots',
        'show_coverage_map': 'Show coverage map',
        'step1_addresses': '1) Addresses',
        'step2_cleanup_optional': '2) Clean up (optional)',
        'reconcile_expander': 'OCR → routing reconciliation',
        'reconcile_col_use': 'Use',
        'reconcile_col_captured': 'Captured',
        'reconcile_col_final': 'Final (editable)',
        'reconcile_col_note': 'Note',
        'apply_to_input': 'Apply to input',
        'use_for_this_run_only': 'Use for this run only',
        'reconcile_tip_apply_persists': "Tip: 'Apply' updates the textarea so your edits persist.",
        'step3_optimize': '3) Optimize',
        'step4_results': '4) Results',
        'reconcile_note_home_excluded': 'Looks like home address (excluded)',
        'reconcile_note_near_duplicate': 'Near-duplicate of row {row} (excluded)',

        # --- camera_ocr_widget additions / alignment ---
        'camera_focus_help': 'Having trouble focusing?',
        'camera_focus_help_bullets': (
            '- Tap on the text to focus/expose.\n'
            '- Add more light (lamp / window) and avoid glare.\n'
            '- Move the phone a little farther away, then slowly closer.\n'
            '- Try 2× zoom instead of moving very close (helps minimum focus distance).\n'
            '- Hold still for 1–2 seconds after tapping to let autofocus settle.\n'
        ),
        'camera_ocr_upload_fallback': 'Or upload a photo',
        'retake': 'Retake photo',

        # These keys are used by the pasted widget; keep your existing ones too.
        'ocr_no_addresses': 'No addresses found.',
        'ocr_debug': 'OCR debug',
        'dropbox_path': 'Dropbox path: {path}',
        'ocr_loaded_n': 'Loaded {n} addresses into the input box.',

        # Backward-compatible aliases (if other code still uses the older keys you already had)
        'ocr_none_found': 'No addresses found.',
        'ocr_debug_title': 'OCR debug',
        'debug_dropbox_path': 'Dropbox path: {path}',
        'ocr_loaded_addresses_ok': 'Loaded {count} addresses into the input box.',
    }
)

TRANSLATIONS['nl'].update(
    {
        # --- app.py additions ---
        'input_preview_caption': 'Voorbeeld invoer (na thuis + ontdubbelen)',
        'col_num': '#',
        'col_address': 'Adres',
        'col_lat': 'Lat',
        'col_lon': 'Lon',
        'col_snap_dist_km': 'Snap-afstand (km)',
        'max_snap_distance_caption': 'Maximale snap-afstand: {km} km',
        'geocoded_title': 'Geocoding en snapping',
        'snapping_overview_title': 'Snapping-overzicht',
        'snap_dist_km_col': 'Snap-afstand (km)',
        'order_title_simple': 'Geoptimaliseerde volgorde',
        'dist_matrix_expander': 'Afstandsmatrix',
        'maps_plots_expander': 'Kaarten / plots',
        'show_coverage_map': 'Toon dekkingskaart',
        'step1_addresses': '1) Adressen',
        'step2_cleanup_optional': '2) Opschonen (optioneel)',
        'reconcile_expander': 'OCR → route-reconciliatie',
        'reconcile_col_use': 'Gebruik',
        'reconcile_col_captured': 'Gevangen',
        'reconcile_col_final': 'Eindtekst (bewerkbaar)',
        'reconcile_col_note': 'Opmerking',
        'apply_to_input': 'Toepassen op invoer',
        'use_for_this_run_only': 'Alleen voor deze run gebruiken',
        'reconcile_tip_apply_persists': "Tip: 'Toepassen' werkt het tekstvak bij zodat je edits blijven staan.",
        'step3_optimize': '3) Optimaliseren',
        'step4_results': '4) Resultaten',
        'reconcile_note_home_excluded': 'Lijkt op thuisadres (uitgesloten)',
        'reconcile_note_near_duplicate': 'Bijna-duplicaat van rij {row} (uitgesloten)',

        # --- camera_ocr_widget additions / alignment ---
        'camera_focus_help': 'Problemen met scherpstellen?',
        'camera_focus_help_bullets': (
            '- Tik op de tekst om te focussen/belichting te zetten.\n'
            '- Zorg voor meer licht (lamp/raam) en vermijd schittering.\n'
            '- Houd de telefoon iets verder weg en ga dan langzaam dichterbij.\n'
            '- Probeer 2× zoom i.p.v. heel dichtbij (helpt bij minimale focusafstand).\n'
            '- Blijf 1–2 seconden stil na het tikken zodat autofocus kan stabiliseren.\n'
        ),
        'camera_ocr_upload_fallback': 'Of upload een foto',
        'retake': 'Foto opnieuw maken',

        'ocr_no_addresses': 'Geen adressen gevonden.',
        'ocr_debug': 'OCR debug',
        'dropbox_path': 'Dropbox-pad: {path}',
        'ocr_loaded_n': '{n} adressen in het invoerveld geladen.',

        # Backward-compatible aliases
        'ocr_none_found': 'Geen adressen gevonden.',
        'ocr_debug_title': 'OCR debug',
        'debug_dropbox_path': 'Dropbox-pad: {path}',
        'ocr_loaded_addresses_ok': '{count} adressen in het invoerveld geladen.',
    }
)

TRANSLATIONS['pt'].update(
    {
        # --- app.py additions ---
        'input_preview_caption': 'Pré-visualização (após casa + remoção de duplicados)',
        'col_num': '#',
        'col_address': 'Endereço',
        'col_lat': 'Lat',
        'col_lon': 'Lon',
        'col_snap_dist_km': 'Dist. de ajuste (km)',
        'max_snap_distance_caption': 'Distância máxima de ajuste: {km} km',
        'geocoded_title': 'Geocodificação e ajuste à rede',
        'snapping_overview_title': 'Resumo do ajuste à rede',
        'snap_dist_km_col': 'Distância de ajuste (km)',
        'order_title_simple': 'Ordem otimizada',
        'dist_matrix_expander': 'Matriz de distâncias',
        'maps_plots_expander': 'Mapas / gráficos',
        'show_coverage_map': 'Mostrar mapa de cobertura',
        'step1_addresses': '1) Endereços',
        'step2_cleanup_optional': '2) Limpeza (opcional)',
        'reconcile_expander': 'Reconciliação OCR → rota',
        'reconcile_col_use': 'Usar',
        'reconcile_col_captured': 'Capturado',
        'reconcile_col_final': 'Final (editável)',
        'reconcile_col_note': 'Nota',
        'apply_to_input': 'Aplicar à entrada',
        'use_for_this_run_only': 'Usar apenas nesta execução',
        'reconcile_tip_apply_persists': "Dica: 'Aplicar' atualiza a caixa de texto para manter as edições.",
        'step3_optimize': '3) Otimizar',
        'step4_results': '4) Resultados',
        'reconcile_note_home_excluded': 'Parece o endereço de casa (excluído)',
        'reconcile_note_near_duplicate': 'Quase duplicado da linha {row} (excluído)',

        # --- camera_ocr_widget additions / alignment ---
        'camera_focus_help': 'Com dificuldade em focar?',
        'camera_focus_help_bullets': (
            '- Toque no texto para focar/ajustar a exposição.\n'
            '- Adicione mais luz (candeeiro/janela) e evite reflexos.\n'
            '- Afaste um pouco o telemóvel e aproxime lentamente.\n'
            '- Experimente zoom 2× em vez de aproximar demasiado (ajuda na distância mínima de foco).\n'
            '- Mantenha-se imóvel durante 1–2 segundos após tocar para o autofocus estabilizar.\n'
        ),
        'camera_ocr_upload_fallback': 'Ou carregue uma foto',
        'retake': 'Tirar outra foto',

        'ocr_no_addresses': 'Não foram encontrados endereços.',
        'ocr_debug': 'Depuração do OCR',
        'dropbox_path': 'Caminho no Dropbox: {path}',
        'ocr_loaded_n': 'Foram carregados {n} endereços na caixa de texto.',

        # Backward-compatible aliases
        'ocr_none_found': 'Não foram encontrados endereços.',
        'ocr_debug_title': 'Depuração do OCR',
        'debug_dropbox_path': 'Caminho no Dropbox: {path}',
        'ocr_loaded_addresses_ok': 'Foram carregados {count} endereços na caixa de texto.',
    }
)

TRANSLATIONS['en'].update(
    {
        'graph_coverage_title': 'Road network coverage',
        'graph_coverage_subtitle': 'Coverage of the available road network.',
        'graph_coverage_subtitle_roi': 'Coverage of the available road network (ROI: {roi}).',
        'graph_coverage_map_title': 'Coverage map',
        'camera_ocr_expander': 'Camera OCR',
    }
)

TRANSLATIONS['nl'].update(
    {
        'graph_coverage_title': 'Dekking van het wegennet',
        'graph_coverage_subtitle': 'Dekking van het beschikbare wegennet.',
        'graph_coverage_subtitle_roi': 'Dekking van het beschikbare wegennet (ROI: {roi}).',
        'graph_coverage_map_title': 'Dekkingskaart',
        'camera_ocr_expander': 'Camera OCR',
    }
)

TRANSLATIONS['pt'].update(
    {
        'graph_coverage_title': 'Cobertura da rede viária',
        'graph_coverage_subtitle': 'Cobertura da rede viária disponível.',
        'graph_coverage_subtitle_roi': 'Cobertura da rede viária disponível (ROI: {roi}).',
        'graph_coverage_map_title': 'Mapa de cobertura',
        'camera_ocr_expander': 'OCR por câmara',
    }
)

