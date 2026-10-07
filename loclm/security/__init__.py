"""LocLM Security Package (V8 Architecture).

Provides offline network enforcement, security policy guards,
pre-execution command/path auditing, and immutable security logs.
"""

from loclm.security.guard import (
    AuditRecord,
    PolicyGuard,
    PolicyViolationError,
    RiskLevel,
    SecurityPolicy,
)
from loclm.security.network import (
    NetworkSecurityError,
    get_network_status,
    is_offline_mode,
    set_offline_mode,
    validate_url,
)

__all__ = [
    "AuditRecord",
    "PolicyGuard",
    "PolicyViolationError",
    "RiskLevel",
    "SecurityPolicy",
    "NetworkSecurityError",
    "get_network_status",
    "is_offline_mode",
    "set_offline_mode",
    "validate_url",
]
