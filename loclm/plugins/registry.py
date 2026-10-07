"""Plugin Registry for LocLM V7.

Integrates discovered plugins with the ToolRegistry and AgentMap,
making plugin tools and agents available to the V7Orchestrator at runtime.
"""

from __future__ import annotations

import logging
from typing import Any

from loclm.agents.base import AgentRole, BaseAgent
from loclm.models.manager import ModelManager
from loclm.plugins.loader import PluginLoader, PluginManifest
from loclm.tools.base import BaseTool
from loclm.tools.registry import ToolRegistry
from loclm.tools.security import SecurityGuard

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Discovers plugins and integrates them with the live tool & agent registries.

    Typical usage (inside V7Orchestrator init)::

        plugin_registry = PluginRegistry(plugin_dir="~/.loclm/plugins")
        plugin_registry.load_and_register(
            tool_registry=self.tool_registry,
            model_manager=self.model_manager,
            agent_map=self._agents,
        )
    """

    def __init__(self, plugin_dir: str | None = None) -> None:
        self._loader = PluginLoader(plugin_dir)
        self._manifests: list[PluginManifest] = []

    @property
    def manifests(self) -> list[PluginManifest]:
        """All loaded plugin manifests."""
        return self._manifests

    @property
    def loaded_count(self) -> int:
        """Number of successfully loaded plugins."""
        return sum(1 for m in self._manifests if m.is_valid)

    def load_and_register(
        self,
        tool_registry: ToolRegistry,
        model_manager: ModelManager,
        agent_map: dict[AgentRole, BaseAgent],
        guard: SecurityGuard | None = None,
    ) -> dict[str, int]:
        """Discover plugins and register all tools and agents.

        Args:
            tool_registry: Active ToolRegistry to register plugin tools into.
            model_manager: ModelManager passed to plugin agent constructors.
            agent_map: Live agent map that plugin agents are inserted into.
            guard: Optional security guard to pass to plugin tools.

        Returns:
            Summary dict with counts: ``{"plugins": N, "tools": N, "agents": N}``.
        """
        self._manifests = self._loader.discover()
        _guard = guard or SecurityGuard()
        tools_registered = 0
        agents_registered = 0

        for manifest in self._manifests:
            if not manifest.is_valid:
                continue

            # Register plugin tools
            for tool_cls in manifest.tools:
                try:
                    instance: BaseTool = self._instantiate_tool(tool_cls, _guard)
                    tool_registry.register_tool(instance)
                    tools_registered += 1
                    logger.info(
                        "Plugin tool registered: %s (from %s)",
                        instance.name,
                        manifest.plugin_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to register plugin tool %s: %s",
                        tool_cls.__name__,
                        exc,
                    )

            # Register plugin agents (mapped to GENERAL role by default unless agent.role exists)
            for agent_cls in manifest.agents:
                try:
                    agent_instance: BaseAgent = self._instantiate_agent(
                        agent_cls, model_manager, tool_registry
                    )
                    role = getattr(agent_instance, "role", AgentRole.GENERAL)
                    agent_map[role] = agent_instance
                    agents_registered += 1
                    logger.info(
                        "Plugin agent registered: %s → role=%s (from %s)",
                        agent_cls.__name__,
                        role,
                        manifest.plugin_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to register plugin agent %s: %s",
                        agent_cls.__name__,
                        exc,
                    )

        logger.info(
            "Plugin loading complete: %d plugin(s), %d tool(s), %d agent(s)",
            self.loaded_count,
            tools_registered,
            agents_registered,
        )
        return {
            "plugins": self.loaded_count,
            "tools": tools_registered,
            "agents": agents_registered,
        }

    def get_plugin_summary(self) -> list[str]:
        """Return human-readable lines for each loaded plugin."""
        return [m.summary() for m in self._manifests]

    @staticmethod
    def _instantiate_tool(tool_cls: type[BaseTool], guard: SecurityGuard) -> BaseTool:
        """Attempt to construct a plugin tool (with or without a guard arg)."""
        try:
            return tool_cls(guard)  # type: ignore[call-arg]
        except TypeError:
            return tool_cls()  # type: ignore[call-arg]

    @staticmethod
    def _instantiate_agent(
        agent_cls: type[BaseAgent],
        model_manager: ModelManager,
        tool_registry: ToolRegistry,
    ) -> BaseAgent:
        """Attempt to construct a plugin agent (various constructor signatures)."""
        try:
            return agent_cls(model_manager, tool_registry)  # type: ignore[call-arg]
        except TypeError:
            try:
                return agent_cls(model_manager)  # type: ignore[call-arg]
            except TypeError:
                return agent_cls()  # type: ignore[call-arg]
