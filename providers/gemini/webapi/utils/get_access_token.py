import re
from typing import NamedTuple

from curl_cffi import CurlFollow, CurlHttpVersion
from curl_cffi.curl import CurlError
from curl_cffi.requests import AsyncSession, BrowserTypeLiteral, Cookies, Response
from curl_cffi.requests.exceptions import (
    ConnectionError as CurlConnectionError,
)
from curl_cffi.requests.exceptions import (
    HTTPError,
)
from curl_cffi.requests.exceptions import (
    Timeout as CurlTimeout,
)

from providers.gemini.webapi.constants import BROWSER_TYPE, Endpoint, Headers, format_http_version
from providers.gemini.webapi.exceptions import (
    AuthError,
    TemporarilyBlockedError,
)
from providers.gemini.webapi.exceptions import (
    TimeoutError as GeminiTimeoutError,
)

from .logger import logger


class InitSession(NamedTuple):
    """Everything a successful init attempt produced.

    Attributes
    ----------
    access_token: `str | None`
        The "SNlM0e" value. Guest sessions have none.
    build_label: `str | None`
        Frontend build label sent back as the `bl` request parameter.
    session_id: `str | None`
        Frontend session id sent back as the `f.sid` request parameter.
    language: `str | None`
        Account language.
    push_id: `str | None`
        File upload push id.
    client: `curl_cffi.requests.AsyncSession`
        The **live** session that succeeded, so the caller can reuse its TLS connection.
    cookie_source: `str`
        Name of the cookie group that produced this session - "Base Cookies" or "Guest".
        A session is accepted as soon as it yields an access
        token, which an unauthenticated one does too, so the caller needs to know which
        group to blame when the session turns out to be unusable.

    """

    access_token: str | None
    build_label: str | None
    session_id: str | None
    language: str | None
    push_id: str | None
    client: AsyncSession
    cookie_source: str


_DOMAIN_NAME = "google.com"
_COOKIE_DOMAIN = f".{_DOMAIN_NAME}"
_COOKIE_PATH = "/"

_ACCESS_TOKEN_RE = re.compile(r'"SNlM0e":\s*"(.*?)"')
_BUILD_LABEL_RE = re.compile(r'"cfb2h":\s*"(.*?)"')
_SESSION_ID_RE = re.compile(r'"FdrFJe":\s*"(.*?)"')
_LANGUAGE_RE = re.compile(r'"TuX5cc":\s*"(.*?)"')
_PUSH_ID_RE = re.compile(r'"qKIAYe":\s*"(.*?)"')


def _jar_signature(jar: Cookies) -> frozenset[tuple[str, str]]:
    """Build a hashable identity of a cookie jar to avoid sending duplicated requests."""
    return frozenset((str(c.name), str(c.value)) for c in jar.jar)


def _to_jar(base_cookies: dict | Cookies) -> Cookies:
    """Normalize user provided cookies into a `Cookies` jar, dropping expired/empty ones."""
    jar = Cookies()
    if isinstance(base_cookies, Cookies):
        for cookie in base_cookies.jar:
            if cookie.value and not cookie.is_expired():
                jar.set(
                    str(cookie.name),
                    str(cookie.value),
                    domain=cookie.domain,
                    path=cookie.path,
                    secure=cookie.secure,
                )
    else:
        for name, value in base_cookies.items():
            if value:
                jar.set(
                    name,
                    value,
                    domain=_COOKIE_DOMAIN,
                    path=_COOKIE_PATH,
                    secure=True,
                )

    return jar


def _fill_missing(jar: Cookies, extra: Cookies) -> Cookies:
    """Complete `jar` with cookies from `extra` that it doesn't already carry.

    Values already present in `jar` always win, so supplied cookies
    never get overwritten by freshly issued anonymous ones.
    """
    merged = Cookies(jar)
    known = {str(c.name) for c in jar.jar}
    for cookie in extra.jar:
        if str(cookie.name) not in known:
            merged.set(
                str(cookie.name),
                str(cookie.value),
                domain=cookie.domain,
                path=cookie.path,
                secure=cookie.secure,
            )

    return merged


async def _send_request(
    client: AsyncSession, cookies: dict | Cookies, verbose: bool = False
) -> Response:
    """Send http request with provided cookies using a shared session."""
    client.cookies.clear()
    if isinstance(cookies, Cookies):
        client.cookies.update(cookies)
    else:
        for k, v in cookies.items():
            client.cookies.set(k, v, domain=_COOKIE_DOMAIN, secure=True)

    response = await client.get(Endpoint.INIT, headers=Headers.GEMINI.value)
    if verbose:
        logger.debug(
            f"HTTP Request: GET {Endpoint.INIT} [{response.status_code}] (HTTP/{format_http_version(response.http_version)})"
        )
    response.raise_for_status()
    return response


def _extract_payload(
    response: Response,
) -> tuple[str | None, str | None, str | None, str | None, str | None] | None:
    """Extract init values from an init response, or `None` if the page carries none of them."""
    access_token = _ACCESS_TOKEN_RE.search(response.text)
    build_label = _BUILD_LABEL_RE.search(response.text)
    session_id = _SESSION_ID_RE.search(response.text)
    language = _LANGUAGE_RE.search(response.text)
    push_id = _PUSH_ID_RE.search(response.text)
    if not (access_token or build_label or session_id or language or push_id):
        return None

    return (
        access_token.group(1) if access_token else None,
        build_label.group(1) if build_label else None,
        session_id.group(1) if session_id else None,
        language.group(1) if language else None,
        push_id.group(1) if push_id else None,
    )


async def get_access_token(
    base_cookies: dict | Cookies,
    proxy: str | None = None,
    verbose: bool = False,
    impersonate: BrowserTypeLiteral = BROWSER_TYPE,
    verify: bool = True,
) -> InitSession:
    """Send a get request to gemini.google.com for each group of available cookies and return
    the value of "SNlM0e" as access token on the first successful request.

    Supplied cookies are tried first. If they fail, a preflight request against google.com
    picks up consent/anonymous cookies to complete them, then attempts a guest session.

    Returns the **live** AsyncSession that succeeded so the caller can reuse the same TLS
    connection for subsequent requests, along with the name of the cookie group it came
    from.

    Parameters
    ----------
    base_cookies: `dict | curl_cffi.requests.Cookies`
        Initial cookies to try. Can be a dictionary or a Cookies object.
    proxy: `str`, optional
        Proxy URL.
    verbose: `bool`, optional
        If True, log more details.
    impersonate: `BrowserTypeLiteral`, optional
        Allow to customize client, default to BROWSER_TYPE.
    verify: `bool`, optional
        Whether to verify SSL certificates.

    Returns
    -------
    :class:`InitSession`
        Named tuple of the access token, build label, session id, language, file push id,
        the live `AsyncSession`, and the name of the cookie group that produced it.

    Raises
    ------
    `gemini_webapi.AuthError`
        If all candidate cookie groups failed authentication or yielded no valid init tokens.
    `gemini_webapi.TimeoutError`
        If a request timed out during initialization.
    `gemini_webapi.TemporarilyBlockedError`
        If the client's IP is rate-limited (HTTP 429).
    `curl_cffi.requests.exceptions.HTTPError`
        If Google Gemini returned an unexpected 5xx server error.
    `curl_cffi.requests.exceptions.CurlError`
        If a network or transport connection failure occurred.

    """
    client = AsyncSession(
        impersonate=impersonate,
        proxy=proxy,
        allow_redirects=CurlFollow.SAFE,
        http_version=CurlHttpVersion.NONE,
        verify=verify,
    )

    try:
        # The account database supplies the last verified cookies.
        base_jar = _to_jar(base_cookies)
        cookie_jars_to_test: list[tuple[Cookies, str]] = (
            [(base_jar, "Base Cookies")] if len(base_jar.jar) else []
        )

        # Phase 2: Try every candidate group as-is, without contacting google.com first
        attempts = 0
        tried_jars: set[frozenset[tuple[str, str]]] = set()

        async def try_jars(jars: list[tuple[Cookies, str]]):
            nonlocal attempts
            for jar, group_name in jars:
                signature = _jar_signature(jar)
                if not signature or signature in tried_jars:
                    continue
                tried_jars.add(signature)

                attempts += 1
                try:
                    response = await _send_request(client, jar, verbose=verbose)
                    if payload := _extract_payload(response):
                        if verbose:
                            logger.debug(f"Init attempt ({attempts}) from {group_name} succeeded.")
                        return payload, group_name
                    if verbose:
                        logger.debug(
                            f"Init attempt ({attempts}) from {group_name} returned no init values."
                        )
                except HTTPError as e:
                    status = e.response.status_code if e.response else None
                    if status == 429:
                        raise TemporarilyBlockedError(
                            "Your IP address has been temporarily flagged or blocked by Google (HTTP 429). "
                            "Please try using a proxy, a different network, or wait for a while before retrying."
                        ) from e
                    if status and status >= 500:
                        if verbose:
                            logger.warning(
                                f"Init attempt ({attempts}) from {group_name} failed due to Google server error (HTTP {status})."
                            )
                        raise
                    if verbose:
                        logger.debug(
                            f"Init attempt ({attempts}) from {group_name} failed (HTTP {status})."
                        )
                except (CurlTimeout, TimeoutError) as e:
                    if verbose:
                        logger.warning(
                            f"Init attempt ({attempts}) from {group_name} timed out: {e}"
                        )
                    raise GeminiTimeoutError(
                        f"Request timed out while connecting to {Endpoint.INIT}: {e}"
                    ) from e
                except (CurlConnectionError, CurlError, OSError) as e:
                    if verbose:
                        logger.warning(
                            f"Init attempt ({attempts}) from {group_name} failed due to network error: {e}"
                        )
                    raise

            return None

        if result := await try_jars(cookie_jars_to_test):
            payload, group_name = result
            return InitSession(*payload, client=client, cookie_source=group_name)

        # Phase 3: Fall back to a preflight request for consent/anonymous cookies, then
        # retry the same groups completed with the missing cookies, and finally guest mode
        try:
            # Start from a clean jar, otherwise cookies left over from the failed
            # attempts above would leak into the preflight and guest sessions
            client.cookies.clear()
            response = await client.get(Endpoint.GOOGLE)
            if verbose:
                logger.debug(
                    f"HTTP Request: GET {Endpoint.GOOGLE} [{response.status_code}] (HTTP/{format_http_version(response.http_version)})"
                )
            preflight_cookies = (
                Cookies(client.cookies) if response.status_code == 200 else Cookies()
            )
        except (CurlTimeout, TimeoutError) as e:
            raise GeminiTimeoutError(
                f"Request timed out while connecting to {Endpoint.GOOGLE}: {e}"
            ) from e
        except (CurlConnectionError, CurlError, OSError) as e:
            if not isinstance(e, HTTPError):
                raise
            logger.warning(f"Preflight request to google.com failed: {e}")
            preflight_cookies = Cookies()
        except Exception as e:
            if not cookie_jars_to_test:
                # Nothing else to fall back on, surface the underlying error
                raise

            logger.warning(f"Preflight request to google.com failed: {e}")
            preflight_cookies = Cookies()

        if _jar_signature(preflight_cookies):
            retries = [
                (_fill_missing(jar, preflight_cookies), f"{group_name} + Preflight")
                for jar, group_name in cookie_jars_to_test
            ]
            retries.append((Cookies(preflight_cookies), "Guest"))
            if result := await try_jars(retries):
                payload, group_name = result
                return InitSession(*payload, client=client, cookie_source=group_name)

        raise AuthError(
            f"Failed to initialize client after {attempts} attempts. SECURE_1PSIDTS "
            "could get expired frequently, please make sure cookie values are up to date."
        )
    except BaseException:
        await client.close()
        raise
