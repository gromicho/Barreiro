import os
import dropbox


def make_dropbox_client() -> dropbox.Dropbox:
    """
    Create a Dropbox client using OAuth refresh token flow.

    Expects environment variables:
      - DROPBOX_APP_KEY
      - DROPBOX_APP_SECRET
      - DROPBOX_REFRESH_TOKEN
    """
    app_key = os.environ['DROPBOX_APP_KEY']
    app_secret = os.environ['DROPBOX_APP_SECRET']
    refresh_token = os.environ['DROPBOX_REFRESH_TOKEN']

    return dropbox.Dropbox(
        oauth2_refresh_token=refresh_token,
        app_key=app_key,
        app_secret=app_secret,
    )
