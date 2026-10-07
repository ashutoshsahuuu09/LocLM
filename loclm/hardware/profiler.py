"""Hardware performance profiler.

Classifies the detected hardware into performance tiers
that drive model selection decisions.
"""

from __future__ import annotations

import logging
from enum import IntEnum

from pydantic import BaseModel, Field

from loclm.hardware.detector import HardwareInfo

logger = logging.getLogger(__name__)


class PerformanceTier(IntEnum):
    """Hardware performance tiers for model selection.

    Higher tier = more capable hardware = larger models.
    """

    TIER_0 = 0  # CPU + <8GB RAM -> tiny models
    TIER_1 = 1  # CPU + 8-16GB RAM -> small models
    TIER_2 = 2  # GPU 4-8GB VRAM -> small/medium quantized
    TIER_3 = 3  # GPU 8-16GB VRAM -> medium/large quantized
    TIER_4 = 4  # GPU 16-24GB VRAM -> large models
    TIER_5 = 5  # GPU 24GB+ VRAM -> advanced models

    @property
    def label(self) -> str:
        """Human-readable label for the tier."""
        labels = {
            0: "CPU LOW",
            1: "CPU",
            2: "LOW GPU",
            3: "MID GPU",
            4: "HIGH GPU",
            5: "ULTRA GPU",
        }
        return labels.get(self.value, "UNKNOWN")

    @property
    def description(self) -> str:
        """Description of the tier's capabilities."""
        descriptions = {
            0: "CPU-only with limited RAM -- tiny models, basic Q&A",
            1: "CPU-only with adequate RAM -- small models, simple tasks",
            2: "Entry-level GPU -- small/medium quantized models",
            3: "Mid-range GPU -- medium/large quantized models",
            4: "High-end GPU -- large models, strong capability",
            5: "Top-tier GPU -- advanced models, maximum capability",
        }
        return descriptions.get(self.value, "Unknown tier")


class HardwareProfile(BaseModel):
    """Complete hardware profile with performance tier classification."""

    hardware: HardwareInfo
    tier: PerformanceTier
    max_model_size_gb: float = Field(
        description="Maximum model size in GB that can be loaded"
    )
    recommended_context: int = Field(
        description="Recommended context window size"
    )
    notes: list[str] = Field(default_factory=list, description="Profiling notes")


def profile_hardware(hardware: HardwareInfo) -> HardwareProfile:
    """Classify hardware into a performance tier and compute limits.

    Args:
        hardware: Detected hardware information.

    Returns:
        HardwareProfile with tier, max model size, and recommendations.
    """
    notes: list[str] = []

    # Determine tier based on GPU VRAM, then fall back to RAM
    if hardware.has_gpu and hardware.gpu_vram_gb is not None:
        vram = hardware.gpu_vram_gb

        if vram >= 24:
            tier = PerformanceTier.TIER_5
            max_model_size = vram * 0.85  # Leave 15% headroom
            context = 65536
            notes.append(f"GPU with {vram}GB VRAM -- top-tier capability")
        elif vram >= 16:
            tier = PerformanceTier.TIER_4
            max_model_size = vram * 0.85
            context = 32768
            notes.append(f"GPU with {vram}GB VRAM -- high capability")
        elif vram >= 8:
            tier = PerformanceTier.TIER_3
            max_model_size = vram * 0.80
            context = 16384
            notes.append(f"GPU with {vram}GB VRAM -- mid-range capability")
        elif vram >= 4:
            tier = PerformanceTier.TIER_2
            max_model_size = vram * 0.80
            context = 8192
            notes.append(f"GPU with {vram}GB VRAM -- entry-level GPU mode")
        else:
            # GPU with <4GB VRAM -- treat as CPU-class
            tier = PerformanceTier.TIER_1 if hardware.ram_total_gb >= 8 else PerformanceTier.TIER_0
            max_model_size = hardware.ram_total_gb * 0.4
            context = 4096
            notes.append(f"GPU with only {vram}GB VRAM -- falling back to CPU-class tier")
    else:
        # CPU-only mode
        ram = hardware.ram_total_gb
        if ram >= 8:
            tier = PerformanceTier.TIER_1
            max_model_size = ram * 0.40  # Conservative: 40% of RAM for model
            context = 4096
            notes.append(f"CPU-only mode with {ram}GB RAM")
        else:
            tier = PerformanceTier.TIER_0
            max_model_size = ram * 0.35
            context = 2048
            notes.append(f"CPU-only mode with limited RAM ({ram}GB)")

    # Apple Silicon gets a small boost due to unified memory efficiency
    if hardware.metal_available:
        max_model_size *= 1.1
        notes.append("Apple Metal available -- unified memory boost applied")

    # CUDA present
    if hardware.cuda_available:
        notes.append(f"CUDA {hardware.cuda_version or 'detected'} -- GPU acceleration enabled")

    # ROCm present
    if hardware.rocm_available:
        notes.append("AMD ROCm detected -- GPU acceleration enabled")

    # Storage warning
    if hardware.storage_free_gb < 10:
        notes.append(f"[WARN] Low disk space: {hardware.storage_free_gb}GB free")

    max_model_size = round(max_model_size, 1)

    logger.info(
        "Hardware profiled: tier=%s, max_model=%.1fGB, context=%d",
        tier.label,
        max_model_size,
        context,
    )

    return HardwareProfile(
        hardware=hardware,
        tier=tier,
        max_model_size_gb=max_model_size,
        recommended_context=context,
        notes=notes,
    )
