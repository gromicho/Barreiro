'''
Streamlit UI widgets (thin glue).

This module contains small, composable UI widget functions. It should not contain
routing logic or heavy services—only orchestration of UI + calling services.
'''

from __future__ import annotations

import os

import streamlit as st
from openai import OpenAI

from services.address_ocr import extract_addresses_from_image
from persistence.dropbox_store import save_debug_photo

from ui.drive_handlers import (
    clear_geocoding_cache,
    get_address_versions_for_ui,
    load_addresses_version_from_drive,
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


DEFAULT_OCR_MODEL = 'gpt-4.1-mini'


# -----------------------------------------------------------------------------
# OpenAI client
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_openai_client() -> OpenAI:
    '''
    Create and cache the OpenAI client.

    Returns:
        Cached OpenAI client.

    Raises:
        RuntimeError: If OPENAI_API_KEY is missing.
    '''
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('Missing OPENAI_API_KEY')
    return OpenAI(api_key=api_key)


# -----------------------------------------------------------------------------
# Camera OCR (camera-only) + Dropbox debug storage
# -----------------------------------------------------------------------------

def camera_ocr_widget(
    *,
    filename: str | None = None,
    model: str = DEFAULT_OCR_MODEL,
    overwrite: bool = True,
    show_debug: bool = False,
    duplicate_first_on_overwrite: bool = False,
) -> bool:
    '''
    ... (same docstring, add:)

    Args:
        duplicate_first_on_overwrite: If True and overwrite=True, duplicate the first
            extracted address by prepending it as the first line.

    Returns:
        True if OCR produced addresses and updated state, else False.
    '''
    st.subheader('📷 Camera OCR')

    photo = st.camera_input('Take a photo of the address note')
    if photo is None:
        return False

    image_bytes = photo.getvalue()
    mime_type = photo.type or 'image/jpeg'

    effective_filename = filename or get_store_filename()

    saved_path: str | None = None
    try:
        saved = save_debug_photo(
            image_bytes,
            filename=effective_filename,
            mime_type=mime_type,
            label='camera',
            make_shared_link=False,
        )
        saved_path = str(saved.get('path') or '') or None
        if show_debug and saved_path:
            st.caption(f'Saved debug photo: {saved_path}')
    except Exception as exc:
        st.warning(f'Dropbox photo save failed (continuing): {exc}')

    try:
        client = _get_openai_client()
        with st.spinner('Extracting addresses...'):
            addresses, raw = extract_addresses_from_image(
                client=client,
                image_bytes=image_bytes,
                mime_type=mime_type,
                model=model,
            )
    except Exception as exc:
        st.error(f'OCR failed: {exc}')
        return False

    if not addresses:
        st.info('No addresses found.')
        if show_debug:
            with st.expander('OCR debug'):
                if saved_path:
                    st.write(f'Dropbox path: `{saved_path}`')
                st.code(raw, language='json')
        return False

    if overwrite:
        if duplicate_first_on_overwrite and len(addresses) >= 1:
            addresses = [addresses[0]] + addresses

        new_text = '\n'.join(addresses).strip()
        set_addresses_text(new_text)
    else:
        new_text = '\n'.join(addresses).strip()
        existing = get_addresses_text().strip()
        combined = (existing + '\n' + new_text).strip() if existing else new_text
        set_addresses_text(combined)

    st.success(f'Loaded {len(addresses)} addresses into the input box.')

    if show_debug:
        with st.expander('OCR debug'):
            if saved_path:
                st.write(f'Dropbox path: `{saved_path}`')
            st.code(raw, language='json')

    return True

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


def drive_version_loader(
    *,
    default_text: str,
    width: str = 'stretch',
    rerun_after_load: bool = True,
) -> None:
    """UI control to load a specific saved address version.

    Intended for the "Full" UI only.
    """
    filename = get_store_filename()
    versions = get_address_versions_for_ui(filename=filename)

    if not versions:
        st.caption(t('no_versions'))
        return

    options: list[str] = []
    version_by_label: dict[str, int] = {}

    for item in versions:
        ver = int(item.get('version', 0))
        ts = str(item.get('timestamp', '') or '').strip()

        raw_count = item.get('address_count', None)
        address_count: int | None = raw_count if isinstance(raw_count, int) else None

        parts: list[str] = [f'v{ver}']
        if address_count is not None:
            parts.append(f'{address_count} addresses')
        if ts:
            parts.append(ts)

        label = ' (' + ', '.join(parts[1:]) + ')' if len(parts) > 1 else ''
        full_label = parts[0] + label

        options.append(full_label)
        version_by_label[full_label] = ver

    col_a, col_b = st.columns([3, 1])
    with col_a:
        choice = st.selectbox(t('version_label'), options, index=0, key='addresses_version_select')
    with col_b:
        if st.button(t('load_version'), width=width):
            try:
                ver = version_by_label.get(choice, int(versions[0].get('version', 0)))
                load_addresses_version_from_drive(default_text=default_text, version=int(ver))
                st.success(t('loaded_version_ok', version=int(ver)))
                if rerun_after_load:
                    st.rerun()
            except UiStateError as exc:
                st.error(t('load_version_failed', error=str(exc)))


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
