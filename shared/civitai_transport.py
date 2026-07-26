"""HTTPS-only, CivitAI-host-restricted transport helpers."""

from urllib.parse import urljoin, urlsplit

import requests


CIVITAI_HOST_SUFFIXES = ("civitai.com", "civitai.red")
MAX_REDIRECTS = 5


def validate_civitai_url(url: str) -> str:
    """Return a CivitAI HTTPS URL or reject an unsafe/untrusted destination."""
    if not isinstance(url, str):
        raise ValueError("CivitAI URL must be a string")

    parsed = urlsplit(url)
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    trusted_host = any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in CIVITAI_HOST_SUFFIXES
    )

    if (
        parsed.scheme != "https"
        or not hostname
        or not trusted_host
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("URL must be an HTTPS URL hosted by CivitAI")

    return url


def civitai_request(method: str, url: str, *, headers=None, data=None, timeout: int = 30, stream: bool = False):
    """Request a trusted CivitAI URL and validate every redirect destination."""
    current_url = validate_civitai_url(url)

    for _ in range(MAX_REDIRECTS + 1):
        response = requests.request(
            method,
            current_url,
            headers=headers,
            data=data,
            timeout=timeout,
            stream=stream,
            allow_redirects=False,
        )
        if not response.is_redirect:
            return response

        location = response.headers.get("Location")
        response.close()
        if not location:
            raise requests.TooManyRedirects("CivitAI returned a redirect without a Location header")
        current_url = validate_civitai_url(urljoin(current_url, location))

    raise requests.TooManyRedirects(f"CivitAI request exceeded {MAX_REDIRECTS} redirects")
