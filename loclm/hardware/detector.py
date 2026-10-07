"""Hardware detection module.

Detects OS, CPU, RAM, GPU, VRAM, CUDA, Metal, and storage
across Windows, Linux, and macOS.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from typing import Any

import psutil
from pydantic import BaseModel, Field

from loclm.hardware.gpu import detect_gpu, GPUInfo

logger = logging.getLogger(__name__)


class HardwareInfo(BaseModel):
    """Complete hardware profile of the local machine."""

    # Operating System
    os_name: str = Field(description="Operating system name")
    os_version: str = Field(default="", description="OS version string")
    os_arch: str = Field(default="", description="CPU architecture (x86_64, arm64, etc.)")

    # CPU
    cpu_name: str = Field(default="Unknown CPU", description="CPU model name")
    cpu_cores_physical: int = Field(default=1, description="Physical CPU cores")
    cpu_cores_logical: int = Field(default=1, description="Logical CPU cores (with HT)")

    # Memory
    ram_total_gb: float = Field(default=0.0, description="Total system RAM in GB")
    ram_available_gb: float = Field(default=0.0, description="Available RAM in GB")

    # GPU
    gpu_name: str | None = Field(default=None, description="GPU model name")
    gpu_vram_gb: float | None = Field(default=None, description="GPU VRAM in GB")
    gpu_vram_free_gb: float | None = Field(default=None, description="Free VRAM in GB")
    gpu_driver: str | None = Field(default=None, description="GPU driver version")

    # Acceleration
    cuda_available: bool = Field(default=False, description="CUDA availability")
    cuda_version: str | None = Field(default=None, description="CUDA version")
    metal_available: bool = Field(default=False, description="Apple Metal availability")
    rocm_available: bool = Field(default=False, description="AMD ROCm availability")

    # Storage
    storage_total_gb: float = Field(default=0.0, description="Total disk space in GB")
    storage_free_gb: float = Field(default=0.0, description="Free disk space in GB")

    @property
    def has_gpu(self) -> bool:
        """Check if a usable GPU was detected."""
        return self.gpu_name is not None and self.gpu_vram_gb is not None

    @property
    def has_acceleration(self) -> bool:
        """Check if hardware acceleration is available."""
        return self.cuda_available or self.metal_available or self.rocm_available

    @property
    def effective_vram_gb(self) -> float:
        """Return available VRAM or 0 if no GPU."""
        return self.gpu_vram_gb or 0.0

    @property
    def effective_ram_gb(self) -> float:
        """Return available RAM for model loading (heuristic: 70% of free)."""
        return round(self.ram_available_gb * 0.7, 1)


def detect_hardware() -> HardwareInfo:
    """Detect all hardware capabilities of the local machine.

    Returns:
        HardwareInfo with all detected hardware properties.
    """
    info: dict[str, Any] = {}

    # --- Operating System ---
    info["os_name"] = _detect_os_name()
    info["os_version"] = platform.version()
    info["os_arch"] = platform.machine()

    # --- CPU ---
    info["cpu_name"] = _detect_cpu_name()
    info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or 1
    info["cpu_cores_logical"] = psutil.cpu_count(logical=True) or 1

    # --- RAM ---
    mem = psutil.virtual_memory()
    info["ram_total_gb"] = round(mem.total / (1024**3), 1)
    info["ram_available_gb"] = round(mem.available / (1024**3), 1)

    # --- GPU ---
    gpu_info = detect_gpu()
    if gpu_info:
        info["gpu_name"] = gpu_info.name
        info["gpu_vram_gb"] = gpu_info.vram_total_gb
        info["gpu_vram_free_gb"] = gpu_info.vram_free_gb
        info["gpu_driver"] = gpu_info.driver_version
        info["cuda_available"] = gpu_info.cuda_available
        info["cuda_version"] = gpu_info.cuda_version
        info["rocm_available"] = gpu_info.rocm_available

    # --- Apple Metal ---
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        info["metal_available"] = True

    # --- Storage ---
    try:
        disk = shutil.disk_usage(".")
        info["storage_total_gb"] = round(disk.total / (1024**3), 1)
        info["storage_free_gb"] = round(disk.free / (1024**3), 1)
    except OSError:
        logger.warning("Could not detect disk usage")

    return HardwareInfo(**info)


def _detect_os_name() -> str:
    """Detect a human-readable OS name."""
    system = platform.system()
    if system == "Windows":
        release = platform.release()
        return f"Windows {release}"
    elif system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return f"macOS {mac_ver}" if mac_ver else "macOS"
    elif system == "Linux":
        try:
            # Try to get distribution name
            result = subprocess.run(
                ["cat", "/etc/os-release"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip('"')
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return "Linux"
    return system


def _detect_cpu_name() -> str:
    """Detect CPU model name across platforms."""
    system = platform.system()

    try:
        if system == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip()

        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()

        elif system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.debug("CPU name detection failed: %s", e)

    return platform.processor() or "Unknown CPU"
