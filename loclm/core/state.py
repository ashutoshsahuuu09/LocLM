"""Application state module.

Central state container for LocLM that holds all runtime
components and configuration. Initialized once at startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from loclm.hardware.detector import HardwareInfo, detect_hardware
from loclm.hardware.profiler import HardwareProfile, PerformanceTier, profile_hardware
from loclm.models.base import ModelRuntime
from loclm.models.manager import ModelManager
from loclm.models.ollama import OllamaRuntime
from loclm.models.selector import ModelSelector
from loclm.security.network import set_offline_mode

logger = logging.getLogger(__name__)

# Default config path
_CONFIG_DIR = Path(__file__).parent.parent / ".." / "config"


class AppConfig(BaseModel):
    """Application configuration loaded from config.yaml."""

    # App metadata
    app_name: str = "LocLM"
    app_version: str = "0.1.0"
    app_tagline: str = "Local * Private * Agentic"

    # Privacy
    offline_only: bool = True
    telemetry: bool = False

    # Runtime
    runtime_backend: str = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    request_timeout: float = 120.0
    stream_responses: bool = True

    # Chat
    system_prompt: str = (
        "You are LocLM, a helpful local AI assistant running entirely on the user's computer. "
        "You are privacy-focused and operate fully offline. "
        "Be concise, accurate, and helpful. Keep responses short unless asked for detail."
    )
    max_context_messages: int = 50
    max_tokens: int = 1024
    temperature: float = 0.6

    # Agent loop limits
    max_iterations: int = 25
    max_retries: int = 3
    max_execution_time: int = 300


class AppState:
    """Central application state.

    Holds all initialized components and provides access to them.
    Created once at startup via `initialize()`.
    """

    def __init__(self) -> None:
        self.config: AppConfig = AppConfig()
        self.hardware: HardwareInfo | None = None
        self.profile: HardwareProfile | None = None
        self.runtime: ModelRuntime | None = None
        self.model_manager: ModelManager | None = None
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, config_dir: Path | None = None) -> None:
        """Initialize all LocLM components.

        Order:
        1. Load configuration
        2. Set offline mode
        3. Detect hardware
        4. Profile hardware
        5. Create runtime
        6. Create model manager
        7. Discover and select models

        Args:
            config_dir: Override config directory path.
        """
        config_dir = config_dir or _find_config_dir()

        # 1. Load configuration
        self.config = _load_config(config_dir)
        logger.info("Configuration loaded")

        # 2. Enforce offline mode
        set_offline_mode(self.config.offline_only)

        # 3. Detect hardware
        logger.info("Detecting hardware...")
        self.hardware = detect_hardware()
        logger.info(
            "Hardware: %s, %s, RAM=%.1fGB, GPU=%s",
            self.hardware.os_name,
            self.hardware.cpu_name,
            self.hardware.ram_total_gb,
            self.hardware.gpu_name or "None",
        )

        # 4. Profile hardware
        self.profile = profile_hardware(self.hardware)
        logger.info("Performance tier: %s", self.profile.tier.label)

        # 5. Create runtime
        self.runtime = self._create_runtime()
        logger.info("Runtime: %s", self.runtime.runtime_name)

        # 6. Create model manager
        selector = ModelSelector(config_dir / "models.yaml")
        self.model_manager = ModelManager(
            runtime=self.runtime,
            profile=self.profile,
            selector=selector,
        )

        # 7. Initialize model manager (discover models, auto-select)
        await self.model_manager.initialize()

        self._initialized = True
        logger.info("LocLM initialized successfully")

    async def shutdown(self) -> None:
        """Clean shutdown of all components."""
        if self.runtime and isinstance(self.runtime, OllamaRuntime):
            await self.runtime.close()
        self._initialized = False
        logger.info("LocLM shut down")

    def _create_runtime(self) -> ModelRuntime:
        """Create the appropriate model runtime based on config."""
        backend = self.config.runtime_backend.lower()
        if backend == "ollama":
            return OllamaRuntime(
                host=self.config.ollama_host,
                timeout=self.config.request_timeout,
            )
        raise ValueError(f"Unsupported runtime backend: {backend}")


def _find_config_dir() -> Path:
    """Find the config directory, checking multiple locations."""
    # Check relative to CWD
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists():
        return cwd_config

    # Check relative to package
    pkg_config = Path(__file__).parent.parent.parent / "config"
    if pkg_config.exists():
        return pkg_config

    # Default to CWD/config (will use defaults if files don't exist)
    return cwd_config


def _load_config(config_dir: Path) -> AppConfig:
    """Load application config from YAML file."""
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        logger.info("No config file found at %s, using defaults", config_file)
        return AppConfig()

    try:
        with open(config_file) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        # Flatten nested YAML structure into AppConfig fields
        app = raw.get("app", {})
        privacy = raw.get("privacy", {})
        runtime = raw.get("runtime", {})
        chat = raw.get("chat", {})
        agent = raw.get("agent", {})

        return AppConfig(
            app_name=app.get("name", "LocLM"),
            app_version=app.get("version", "0.1.0"),
            app_tagline=app.get("tagline", "Local * Private * Agentic"),
            offline_only=privacy.get("offline_only", True),
            telemetry=privacy.get("telemetry", False),
            runtime_backend=runtime.get("backend", "ollama"),
            ollama_host=runtime.get("ollama_host", "http://127.0.0.1:11434"),
            request_timeout=runtime.get("request_timeout", 120.0),
            stream_responses=runtime.get("stream_responses", True),
            system_prompt=chat.get("system_prompt", AppConfig.model_fields["system_prompt"].default),
            max_context_messages=chat.get("max_context_messages", 50),
            max_tokens=chat.get("max_tokens", 1024),
            temperature=chat.get("temperature", 0.6),
            max_iterations=agent.get("max_iterations", 25),
            max_retries=agent.get("max_retries", 3),
            max_execution_time=agent.get("max_execution_time", 300),
        )

    except (yaml.YAMLError, OSError) as e:
        logger.error("Failed to load config: %s -- using defaults", e)
        return AppConfig()


# Global singleton -- initialized once at startup
_state: AppState | None = None


def get_state() -> AppState:
    """Get the global AppState singleton."""
    global _state
    if _state is None:
        _state = AppState()
    return _state


def reset_state() -> None:
    """Reset the global state (used in testing)."""
    global _state
    _state = None
