"""Model selector module.

Selects the best model for a given task type based on:
- Hardware performance tier
- Available (installed) models
- Model compatibility with available resources
- Fallback chains
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from loclm.hardware.profiler import HardwareProfile, PerformanceTier
from loclm.models.base import ModelInfo, TaskType

logger = logging.getLogger(__name__)

# Default config path relative to package
_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


class ModelSelector:
    """Selects the optimal model based on hardware tier and available models.

    Uses the models.yaml configuration for tier-to-model mappings
    and fallback chains.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _CONFIG_DIR / "models.yaml"
        self._config: dict = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load model configuration from YAML."""
        try:
            if self._config_path.exists():
                with open(self._config_path) as f:
                    self._config = yaml.safe_load(f) or {}
                logger.debug("Loaded model config from %s", self._config_path)
            else:
                logger.warning("Model config not found at %s, using defaults", self._config_path)
                self._config = self._default_config()
        except (yaml.YAMLError, OSError) as e:
            logger.error("Failed to load model config: %s", e)
            self._config = self._default_config()

    def select_model(
        self,
        task_type: TaskType,
        profile: HardwareProfile,
        available_models: list[ModelInfo],
    ) -> str | None:
        """Select the best model for a task type given hardware and available models.

        Strategy:
        1. Look up the recommended model for this tier + task
        2. Check if it's available (installed)
        3. If not, walk the fallback chain
        4. If nothing in the chain is available, pick any compatible installed model
        5. Return None if nothing works

        Args:
            task_type: The type of task (general, coding, reasoning, etc.)
            profile: Hardware profile with tier info.
            available_models: List of installed models.

        Returns:
            Model name string, or None if no compatible model found.
        """
        if not available_models:
            logger.warning("No models available for selection")
            return None

        available_names = {m.name for m in available_models}
        available_by_name = {m.name: m for m in available_models}
        task_key = task_type.value
        tier_key = f"tier_{profile.tier.value}"

        # Step 1: Try the tier-recommended model
        tiers = self._config.get("tiers", {})
        tier_config = tiers.get(tier_key, {})
        tier_models = tier_config.get("models", {})
        recommended = tier_models.get(task_key)

        if recommended and recommended in available_names:
            model = available_by_name[recommended]
            if self._is_compatible(model, profile):
                logger.info("Selected tier-recommended model: %s", recommended)
                return recommended

        # Step 2: Pick the best compatible installed model
        # Prefer larger models but AVOID slow "thinking" models (qwen3, deepseek-r1)
        # which generate thousands of hidden thinking tokens before answering
        all_compatible = [
            m for m in available_models
            if self._is_compatible(m, profile)
        ]

        if all_compatible:
            best = max(all_compatible, key=lambda m: self._model_score(m))
            logger.info("Selected best compatible model: %s (%.1fGB, score=%.1f)",
                        best.name, best.size_gb, self._model_score(best))
            return best.name

        # Step 4: Last resort -- pick the smallest available model
        if available_models:
            smallest = min(available_models, key=lambda m: m.size_gb)
            logger.warning(
                "No fully compatible model found. Using smallest available: %s",
                smallest.name,
            )
            return smallest.name

        return None

    def get_recommended_models(
        self,
        profile: HardwareProfile,
    ) -> dict[str, str]:
        """Get recommended models for all task types based on hardware tier.

        Returns:
            Dict mapping task type to recommended model name.
        """
        tier_key = f"tier_{profile.tier.value}"
        tiers = self._config.get("tiers", {})
        tier_config = tiers.get(tier_key, {})
        return tier_config.get("models", {})

    def get_model_size_estimate(self, model_name: str) -> float:
        """Get estimated model size in GB from config."""
        sizes = self._config.get("model_sizes", {})
        return sizes.get(model_name, 0.0)

    def _is_compatible(self, model: ModelInfo, profile: HardwareProfile) -> bool:
        """Check if a model can run on the given hardware.

        A model is compatible if its size fits within the max model size
        determined by the hardware profile.
        """
        if model.size_gb <= 0:
            # Unknown size -- assume compatible (Ollama handles this)
            return True
        return model.size_gb <= profile.max_model_size_gb

    @staticmethod
    def _model_score(model: ModelInfo) -> float:
        """Score a model for selection. Higher = better.

        Prefers larger models (more capable) but heavily penalizes
        "thinking" models that generate thousands of hidden reasoning
        tokens before producing visible output. These models can take
        2-5 minutes on limited VRAM hardware.

        Thinking models: qwen3, deepseek-r1, qwq, etc.
        Fast models: llama3.2, phi3, qwen2.5, gemma, mistral, etc.
        """
        # Slow model prefixes -- these have hidden thinking overhead
        SLOW_PREFIXES = ("qwen3", "deepseek-r1", "qwq")

        name_lower = model.name.lower().split(":")[0]
        score = model.size_gb  # Base score = model size (bigger = more capable)

        if any(name_lower.startswith(prefix) for prefix in SLOW_PREFIXES):
            # Heavy penalty -- prefer a smaller fast model over a larger slow one
            score *= 0.1
            logger.debug("Penalized slow/thinking model: %s (score=%.2f)", model.name, score)

        return score

    @staticmethod
    def _default_config() -> dict:
        """Minimal fallback configuration if YAML is missing."""
        return {
            "tiers": {
                "tier_0": {"models": {"general": "tinyllama:latest"}},
                "tier_1": {"models": {"general": "qwen2.5:1.5b", "coding": "qwen2.5-coder:1.5b"}},
                "tier_2": {"models": {"general": "qwen2.5:7b", "coding": "qwen2.5-coder:7b"}},
                "tier_3": {"models": {"general": "qwen2.5:14b", "coding": "qwen2.5-coder:14b"}},
                "tier_4": {"models": {"general": "qwen2.5:32b", "coding": "qwen2.5-coder:32b"}},
                "tier_5": {"models": {"general": "qwen2.5:72b", "coding": "qwen2.5-coder:32b"}},
            },
            "fallback_order": {
                "general": [
                    "qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b",
                    "llama3.2:3b", "llama3.2:1b", "tinyllama:latest",
                ],
                "coding": [
                    "qwen2.5-coder:7b", "qwen2.5-coder:3b", "qwen2.5-coder:1.5b",
                    "qwen2.5:7b", "qwen2.5:3b",
                ],
            },
            "model_sizes": {},
        }
