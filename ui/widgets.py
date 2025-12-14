"""Streamlit UI widgets (thin glue)."""

import streamlit as st

from ui.drive_handlers import (
    clear_geocoding_cache,
    reload_addresses_from_drive,
    save_addresses_to_drive,
)
from ui.errors import UiStateError
from ui.i18n.t import t
from ui.state_accessors import (
    get_addresses_text,
    get_store_filename,
    set_addresses_text,
)
from ui.state_keys import init_state_if_missing


def addresses_text_area(
    *,
    label: str | None = None,
    height: int = 200,
    key: str = 'addresses_text_area',
) -> str:
    """Render the addresses text area and keep session_state in sync."""
    if label is None:
        label = t('addresses_label')

    init_state_if_missing(filename=get_store_filename())

    value = st.text_area(
        label,
        value=get_addresses_text(),
        height=int(height),
        key=key,
    )
    set_addresses_text(value)
    return value


def drive_buttons_row(
    *,
    default_text: str,
    save_label: str | None = None,
    reload_label: str | None = None,
    width: str = 'stretch',
    rerun_after_reload: bool = True,
) -> None:
    """Render Save/Reload buttons for Drive-backed address persistence."""
    if save_label is None:
        save_label = t('save_addresses')
    if reload_label is None:
        reload_label = t('reload_addresses')

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button(save_label, width=width):
            try:
                save_addresses_to_drive()
                st.success(t('saved_ok'))
            except UiStateError as exc:
                st.error(t('save_failed', error=str(exc)))

    with col_b:
        if st.button(reload_label, width=width):
            try:
                reload_addresses_from_drive(default_text=default_text)
                st.success(t('reloaded_ok'))
                if rerun_after_reload:
                    st.rerun()
            except UiStateError as exc:
                st.error(t('reload_failed', error=str(exc)))


def clear_geocoding_cache_button(
    *,
    label: str | None = None,
    filename: str = 'capelle_addresses.json',
    width: str = 'stretch',
) -> None:
    """Button that clears geocoding_cache in the Drive store."""
    if label is None:
        label = t('clear_cache')

    if st.button(label, width=width):
        try:
            clear_geocoding_cache(filename=filename)
            st.success(t('cache_cleared_ok'))
        except UiStateError as exc:
            st.error(t('clear_failed', error=str(exc)))
