"""Plugin manifest and loader for LocLM V7 Plugin System.

LocLM V7 supports runtime plugin loading: drop a Python file into the
plugins/ directory (default: ~/.loclm/plugins/) and LocLM will automatically
discover and register any BaseTool subclasses and BaseAgent subclasses
defined there — no restart or code change needed.

Plugin discovery rules:
  - Files ending in ``_plugin.py`` are scanned.
  - Any class inheriting from ``BaseTool`` is registered as a tool.
  - Any class inheriting from ``BaseAgent`` is registered as a plugin agent.
  - Each plugin file may optionally define a module-level ``PLUGIN_METADATA``
    dict with keys: ``name``, ``version``, ``description``, ``author``.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from loclm.agents.base import BaseAgent
from loclm.tools.base import BaseTool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PluginManifest:
    """Metadata and contents discovered from a single plugin file."""

    plugin_id: str
    """Unique identifier derived from the plugin filename stem."""

    file_path: Path
    """Absolute path to the plugin source file."""

    name: str = ""
    """Human-readable plugin name (from PLUGIN_METADATA or filename)."""

    version: str = "1.0.0"
    """Plugin version string."""

    description: str = ""
    """Short description of the plugin's purpose."""

    author: str = "community"
    """Plugin author or contributor name."""

    tools: list[type[BaseTool]] = field(default_factory=list)
    """BaseTool subclasses discovered in this plugin."""

    agents: list[type[BaseAgent]] = field(default_factory=list)
    """BaseAgent subclasses discovered in this plugin."""

    load_error: str | None = None
    """Non-None if the plugin failed to load, containing the error message."""

    @property
    def is_valid(self) -> bool:
        """True if plugin loaded successfully."""
        return self.load_error is None

    def summary(self) -> str:
        """Human-readable summary string for display."""
        status = "OK" if self.is_valid else "FAIL"
        return (
            f"{status} [{self.plugin_id}] {self.name} v{self.version} — "
            f"{len(self.tools)} tool(s), {len(self.agents)} agent(s)"
        )


class PluginLoader:
    """Discovers and loads LocLM plugins from a directory.

    Usage::

        loader = PluginLoader(plugin_dir=Path.home() / ".loclm" / "plugins")
        manifests = loader.discover()
        for m in manifests:
            print(m.summary())
    """

    PLUGIN_SUFFIX = "_plugin.py"

    def __init__(self, plugin_dir: str | Path | None = None) -> None:
        if plugin_dir is None:
            plugin_dir = Path.home() / ".loclm" / "plugins"
        self.plugin_dir = Path(plugin_dir)

    def discover(self) -> list[PluginManifest]:
        """Scan plugin_dir for *_plugin.py files and load each.

        Returns:
            List of PluginManifest objects (valid or errored).
        """
        if not self.plugin_dir.exists():
            logger.debug("Plugin directory does not exist: %s", self.plugin_dir)
            return []

        manifests: list[PluginManifest] = []
        plugin_files = sorted(self.plugin_dir.glob(f"*{self.PLUGIN_SUFFIX}"))

        if not plugin_files:
            logger.debug("No plugin files found in %s", self.plugin_dir)
            return []

        logger.info(
            "Discovering %d plugin file(s) in %s",
            len(plugin_files),
            self.plugin_dir,
        )
        for pf in plugin_files:
            manifest = self._load_plugin_file(pf)
            manifests.append(manifest)
            if manifest.is_valid:
                logger.info("Loaded plugin: %s", manifest.summary())
            else:
                logger.warning(
                    "Plugin load failed [%s]: %s", manifest.plugin_id, manifest.load_error
                )

        return manifests

    def _load_plugin_file(self, path: Path) -> PluginManifest:
        """Load a single plugin file and extract tool/agent classes."""
        plugin_id = path.stem  # e.g., "web_search_plugin"
        manifest = PluginManifest(
            plugin_id=plugin_id,
            file_path=path,
            name=plugin_id.replace("_plugin", "").replace("_", " ").title(),
        )

        try:
            module = self._import_module(path, plugin_id)
        except Exception as exc:
            manifest.load_error = str(exc)
            return manifest

        # Read optional PLUGIN_METADATA
        meta: dict[str, Any] = getattr(module, "PLUGIN_METADATA", {})
        manifest.name = meta.get("name", manifest.name)
        manifest.version = meta.get("version", manifest.version)
        manifest.description = meta.get("description", manifest.description)
        manifest.author = meta.get("author", manifest.author)

        # Discover BaseTool subclasses (excluding the abstract base itself)
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue  # skip imported classes
            if issubclass(obj, BaseTool) and obj is not BaseTool:
                manifest.tools.append(obj)
            elif issubclass(obj, BaseAgent) and obj is not BaseAgent:
                manifest.agents.append(obj)

        return manifest

    @staticmethod
    def _import_module(path: Path, module_name: str) -> ModuleType:
        """Dynamically import a Python file as a module."""
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
