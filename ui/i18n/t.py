"""Translation helper."""

from ui.i18n.state import get_language
from ui.i18n.translations import TRANSLATIONS


def t(key: str, **kwargs: object) -> str:
    """Translate a UI string key using the active language.

    Falls back to Dutch, then to the key itself.

    Args:
        key: Translation key.
        **kwargs: Optional format arguments.

    Returns:
        Translated string.
    """
    lang = get_language()
    text = TRANSLATIONS.get(lang, {}).get(key)
    if text is None:
        text = TRANSLATIONS.get('nl', {}).get(key, key)

    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text

    return text
