"""Network security and offline enforcement module.

Ensures LocLM never makes unauthorized external network calls.
Only localhost connections (to Ollama) are permitted.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Only these hosts are allowed for local runtime communication
ALLOWED_HOSTS: frozenset[str] = frozenset({
    "127.0.0.1",
    "localhost",
    "::1",
    "[::1]",
    "0.0.0.0",
})

# Default offline mode
_offline_mode: bool = True


def is_offline_mode() -> bool:
    """Check if offline mode is active."""
    return _offline_mode


def set_offline_mode(enabled: bool) -> None:
    """Set the offline mode flag."""
    global _offline_mode
    _offline_mode = enabled
    if enabled:
        logger.info("[LOCKED] OFFLINE MODE enabled -- all external network access blocked")
    else:
        logger.warning("[WARN] OFFLINE MODE disabled -- external network access permitted")


def validate_url(url: str) -> bool:
    """Validate that a URL points to an allowed (local) host.

    Args:
        url: The URL to validate.

    Returns:
        True if the URL is allowed, False otherwise.

    Raises:
        NetworkSecurityError: If offline mode is active and the URL is external.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if hostname in ALLOWED_HOSTS:
        return True

    if _offline_mode:
        logger.error(
            "BLOCKED: Attempted connection to external host '%s'. "
            "LocLM is running in OFFLINE MODE.",
            hostname,
        )
        raise NetworkSecurityError(
            f"This operation requires internet access.\n"
            f"LocLM is currently running in Offline Mode.\n"
            f"Blocked host: {hostname}"
        )

    logger.warning("External network access to '%s' -- offline mode is disabled", hostname)
    return True


def validate_ollama_url(url: str) -> bool:
    """Validate that an Ollama URL is local.

    Args:
        url: The Ollama API URL to validate.

    Returns:
        True if the URL is a valid local Ollama endpoint.
    """
    return validate_url(url)


class NetworkSecurityError(Exception):
    """Raised when a network operation violates offline mode policy."""

    pass


def get_network_status() -> dict[str, str | bool]:
    """Return current network security status for display."""
    return {
        "offline_mode": _offline_mode,
        "network": "BLOCKED" if _offline_mode else "PERMITTED",
        "inference": "LOCAL" if _offline_mode else "LOCAL + REMOTE",
        "memory": "LOCAL",
        "telemetry": False,
    }
