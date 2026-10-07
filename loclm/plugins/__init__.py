"""LocLM Plugin System (V7).

Provides runtime plugin discovery, loading, and registration of
user-defined tools and agents from local plugin directories.
"""

from loclm.plugins.loader import PluginLoader, PluginManifest
from loclm.plugins.registry import PluginRegistry

__all__ = [
    "PluginLoader",
    "PluginManifest",
    "PluginRegistry",
]
