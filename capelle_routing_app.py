from __future__ import annotations

from routing.app import RoutingAppConfig, run_routing_app


def main() -> None:
    """Capelle entry point."""
    cfg = RoutingAppConfig(
        store_filename='capelle_addresses.json',
        drive_prefix='capelle_drive',
        title_name='Joaquim Gromicho',
        title_city='Capelle aan den IJssel',
    )
    run_routing_app(cfg=cfg)


if __name__ == '__main__':
    main()
