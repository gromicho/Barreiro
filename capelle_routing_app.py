from __future__ import annotations

from routing.app import RoutingAppConfig, run_routing_app


def main() -> None:
    '''Capelle entry point.'''
    cfg = RoutingAppConfig(
        store_filename='capelle_addresses.json',
        drive_prefix='capelle_drive',
        title_name='Joaquim Gromicho',
        title_city='Capelle aan den IJssel',
        home_address='Spoorlaan 6, 2908BG Capelle aan den IJssel, Netherlands',
    )
    run_routing_app(cfg=cfg)


if __name__ == '__main__':
    main()
