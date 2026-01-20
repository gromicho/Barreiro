"""
Streamlit UI widgets (thin glue).

This module contains small, composable UI widget functions. It should not contain
routing logic or heavy services, only orchestration of UI and calling services.
"""

from __future__ import annotations

import os

import streamlit as st
from openai import OpenAI

from persistence.dropbox_store import save_debug_photo
from services.address_ocr import extract_addresses_from_image
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


@st.cache_resource(show_spinner=False)
def _get_openai_client() -> OpenAI:
    """
    Create and cache the OpenAI client.

    Returns:
        Cached OpenAI client.

    Raises:
        RuntimeError: If OPENAI_API_KEY is missing.
    """
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('Missing OPENAI_API_KEY')
    return OpenAI(api_key=api_key)


def camera_ocr_widget(
    *,
    filename: str | None = None,
    model: str = DEFAULT_OCR_MODEL,
    home_address: str | None = None,
    overwrite: bool = True,
    show_debug: bool = False,
    duplicate_first_on_overwrite: bool = False,
    key_prefix: str = 'camera_ocr',
) -> bool:
    '''
    Mobile-friendly camera OCR widget.

    Hides the Streamlit camera preview after a photo is captured (better UX on phones),
    and shows the captured image full-width instead.

    Args:
        filename: Store filename for persistence/debug.
        model: OCR model identifier.
        home_address: Optional home address (currently unused here).
        overwrite: Whether to overwrite addresses state on apply.
        show_debug: Whether to show debug expander with raw OCR output.
        duplicate_first_on_overwrite: Whether to duplicate first extracted line when overwriting.
        key_prefix: Prefix to namespace Streamlit widget keys and session keys.

    Returns:
        True if addresses were applied to state, else False.
    '''
    _ = home_address

    def k(name: str) -> str:
        '''Build a stable namespaced key for Streamlit session/widget state.'''
        return f'{key_prefix}.{name}'

    st.subheader(t('camera_ocr_title'))

    # This flag controls whether we keep showing the camera widget after capture.
    if k('hide_camera_after_capture') not in st.session_state:
        st.session_state[k('hide_camera_after_capture')] = True

    hide_camera_after_capture = bool(st.session_state.get(k('hide_camera_after_capture'), True))

    # If we already have an image cached in session_state, do NOT show the camera widget again.
    # This prevents the narrow preview from staying on screen on mobile.
    image_bytes: bytes | None = st.session_state.get(k("image_bytes"))
    mime_type: str = st.session_state.get(k("mime_type"), "image/jpeg")

    photo = None
    if not (hide_camera_after_capture and image_bytes):
        photo = st.camera_input(t('camera_ocr_take_photo'), key=k('camera_input'))

    if photo is None and image_bytes is None:
        return False

    if photo is not None:
        image_bytes = photo.getvalue()
        mime_type = photo.type or 'image/jpeg'
        st.session_state[k('image_bytes')] = image_bytes
        st.session_state[k('mime_type')] = mime_type

    # Show captured photo in full width
    if image_bytes is not None:
        st.image(image_bytes, use_container_width=True)

    # Allow retake (shows the camera widget again)
    retake_clicked = st.button(t('retake') if 'retake' in t.__code__.co_consts else 'Retake photo', width='stretch', key=k('retake_btn'))
    if retake_clicked:
        for name in ('image_bytes', 'mime_type', 'last_key', 'saved_path', 'raw', 'addresses', 'preview_text'):
            st.session_state.pop(k(name), None)
        st.rerun()
        return False

    # From here on, we must have image_bytes
    assert image_bytes is not None

    cache_key = (len(image_bytes), mime_type)
    if st.session_state.get(k('last_key')) != cache_key:
        st.session_state[k('last_key')] = cache_key

        effective_filename = filename or get_store_filename()

        try:
            saved = save_debug_photo(
                image_bytes,
                filename=effective_filename,
                mime_type=mime_type,
                label='camera',
                make_shared_link=False,
            )
            saved_path = str(saved.get('path') or '') or None
            st.session_state[k('saved_path')] = saved_path
        except Exception as exc:
            st.warning(t('dropbox_photo_save_failed', error=str(exc)))
            st.session_state[k('saved_path')] = None

        raw: str = ''
        try:
            client = _get_openai_client()
            with st.spinner(t('ocr_extracting')):
                addresses, raw = extract_addresses_from_image(
                    client=client,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    model=model,
                )
        except Exception as exc:
            st.error(t('ocr_failed', error=str(exc)))
            return False

        st.session_state[k('raw')] = raw
        st.session_state[k('addresses')] = list(addresses or [])

        if not addresses:
            st.info(t('ocr_no_addresses'))
            if show_debug:
                with st.expander(t('ocr_debug')):
                    saved_path = st.session_state.get(k('saved_path'))
                    if saved_path:
                        st.write(t('dropbox_path', path=saved_path))
                    st.code(raw, language='json')
            return False

        lines = list(addresses)
        if overwrite and duplicate_first_on_overwrite and lines:
            lines = [lines[0]] + lines

        st.session_state[k('preview_text')] = '\n'.join(lines).strip()

    st.markdown('---')
    preview = st.text_area(
        label=t('addresses_label'),
        value=str(st.session_state.get(k('preview_text'), '')),
        height=200,
        key=k('preview_text_area'),
    )

    apply_clicked = st.button(t('apply'), width='stretch', key=k('apply_btn'))
    cancel_clicked = st.button(t('cancel'), width='stretch', key=k('cancel_btn'))

    if apply_clicked:
        text_to_apply = preview.strip()
        if overwrite:
            set_addresses_text(text_to_apply)
        else:
            existing = get_addresses_text().strip()
            combined = (existing + '\n' + text_to_apply).strip() if existing else text_to_apply
            set_addresses_text(combined)

        st.success(t('ocr_loaded_n', n=len([ln for ln in text_to_apply.splitlines() if ln.strip()])))

        for name in (
            'hide_camera_after_capture',
            'image_bytes',
            'mime_type',
            'last_key',
            'saved_path',
            'raw',
            'addresses',
            'preview_text',
        ):
            st.session_state.pop(k(name), None)

        st.rerun()
        return True

    if cancel_clicked:
        for name in (
            'hide_camera_after_capture',
            'image_bytes',
            'mime_type',
            'last_key',
            'saved_path',
            'raw',
            'addresses',
            'preview_text',
        ):
            st.session_state.pop(k(name), None)

        st.rerun()
        return False

    if show_debug:
        with st.expander(t('ocr_debug')):
            saved_path = st.session_state.get(k('saved_path'))
            if saved_path:
                st.write(t('dropbox_path', path=saved_path))
            st.code(str(st.session_state.get(k('raw'), '')), language='json')

    return False


def addresses_text_area(
    *,
    label: str | None = None,
    height: int = 200,
    key: str = 'addresses_text_area',
) -> str:
    """
    Render the addresses text area and keep session_state in sync.

    Args:
        label: Optional label override.
        height: Text area height in pixels.
        key: Streamlit session_state key for the widget.

    Returns:
        Current text area value.
    """
    if label is None:
        label = t('addresses_label')

    init_state_if_missing(filename=get_store_filename())
    st.session_state.setdefault(key, get_addresses_text())

    value = st.text_area(
        label,
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
    """
    Render Save/Reload buttons for Drive-backed address persistence.

    Args:
        default_text: Text to restore if Drive storage is empty.
        save_label: Optional button label override for Save.
        reload_label: Optional button label override for Reload.
        width: Streamlit button width.
        rerun_after_reload: If True, rerun after reload.
    """
    _ = default_text  # used inside handlers, kept for API symmetry
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
    """
    UI control to load a specific saved address version.

    Intended for the "Full" UI only.

    Args:
        default_text: Text to restore if storage is empty.
        width: Streamlit button width.
        rerun_after_load: If True, rerun after load.
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

        label_suffix = ' (' + ', '.join(parts[1:]) + ')' if len(parts) > 1 else ''
        full_label = parts[0] + label_suffix

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
    """
    Button that clears geocoding_cache in the Drive store.

    Args:
        label: Optional label override.
        filename: Store filename whose cache should be cleared.
        width: Streamlit button width.
    """
    if label is None:
        label = t('clear_cache')

    if st.button(label, width=width):
        try:
            clear_geocoding_cache(filename=filename)
            st.success(t('cache_cleared_ok'))
        except UiStateError as exc:
            st.error(t('clear_failed', error=str(exc)))
