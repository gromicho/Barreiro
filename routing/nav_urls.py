# routing/nav_urls.py

import urllib.parse


def build_google_maps_url_from_addresses(addresses: list[str]) -> str:
    """
    Build a Google Maps directions URL with optional waypoints.

    Args:
        addresses: Ordered list of addresses.

    Returns:
        URL string.
    """
    if len(addresses) < 2:
        raise ValueError('At least origin and destination are required.')

    origin = urllib.parse.quote(addresses[0])
    destination = urllib.parse.quote(addresses[-1])
    intermediates = addresses[1:-1]

    base = 'https://www.google.com/maps/dir/?api=1'
    url = f'{base}&origin={origin}&destination={destination}'

    if intermediates:
        waypoints = '|'.join(urllib.parse.quote(a) for a in intermediates)
        url += f'&waypoints={waypoints}'

    return url
