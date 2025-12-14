"""Widgets related to language selection."""

import streamlit as st

from ui.i18n.state import get_language, init_language_if_missing, set_language
from ui.i18n.translations import SUPPORTED_LANGS
from ui.i18n.t import t


def language_selector(*, default_lang: str | None = None, key: str = 'language_selector') -> str:
    """Render a language selector and store choice in session_state.

    Args:
        default_lang: Default language code, or None to auto-detect from browser locale.
        key: Streamlit widget key.

    Returns:
        Selected language code.
    """
    init_language_if_missing(default_lang=default_lang)

    options = list(SUPPORTED_LANGS.keys())
    labels = [SUPPORTED_LANGS[code] for code in options]

    current = get_language()
    try:
        current_idx = options.index(current)
    except ValueError:
        current_idx = 0

    chosen_label = st.selectbox(
        t('language_label'),
        options=labels,
        index=current_idx,
        key=key,
    )
    chosen_code = options[labels.index(chosen_label)]
    set_language(chosen_code)
    return chosen_code
