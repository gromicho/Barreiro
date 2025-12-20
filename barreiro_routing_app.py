from __future__ import annotations

from routing_app import RoutingAppConfig, run_routing_app


def main() -> None:
    """Barreiro entry point."""
    cfg = RoutingAppConfig(
        store_filename='barreiro_addresses.json',
        drive_prefix='barreiro_drive',
        title_name='Dra. Anneke Joosten',
        title_city='Barreiro',
    )
    run_routing_app(cfg=cfg)


if __name__ == '__main__':
    main()
