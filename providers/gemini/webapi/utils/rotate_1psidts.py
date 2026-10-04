from curl_cffi.requests import AsyncSession, Cookies

from providers.gemini.webapi.constants import Endpoint, Headers, format_http_version
from providers.gemini.webapi.exceptions import AuthError

from .logger import logger


def _extract_cookie_value(cookies: Cookies, name: str) -> str | None:
    return next((cookie.value for cookie in cookies.jar if cookie.name == name), None)


async def rotate_1psidts(client: AsyncSession, verbose: bool = False) -> str | None:
    """Rotate the in-memory cookie; the account service verifies and persists it."""
    old_1psidts = _extract_cookie_value(client.cookies, "__Secure-1PSIDTS")
    response = await client.post(
        url=Endpoint.ROTATE_COOKIES,
        headers=Headers.ROTATE_COOKIES.value,
        data='[000,"-0000000000000000000"]',
    )
    if verbose:
        logger.debug(
            f"HTTP Request: POST {Endpoint.ROTATE_COOKIES} [{response.status_code}] "
            f"(HTTP/{format_http_version(response.http_version)})"
        )
    if response.status_code == 401:
        raise AuthError
    response.raise_for_status()

    new_1psidts = _extract_cookie_value(client.cookies, "__Secure-1PSIDTS")
    if new_1psidts and new_1psidts != old_1psidts:
        return new_1psidts
    logger.warning("RotateCookies did not issue a new __Secure-1PSIDTS; retrying next interval.")
    return None
