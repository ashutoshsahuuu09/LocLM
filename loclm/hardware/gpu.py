"""GPU detection module.

Detects NVIDIA, AMD, and Apple Silicon GPUs with VRAM information.
Uses nvidia-smi for NVIDIA GPUs, rocm-smi for AMD, and system profiler for macOS.
"""

from __future__ import annotations

import logging
import platform
import subprocess

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GPUInfo(BaseModel):
    """Detected GPU information."""

    name: str = Field(description="GPU model name")
    vram_total_gb: float = Field(description="Total VRAM in GB")
    vram_free_gb: float | None = Field(default=None, description="Free VRAM in GB")
    vram_used_gb: float | None = Field(default=None, description="Used VRAM in GB")
    driver_version: str | None = Field(default=None, description="GPU driver version")
    cuda_available: bool = Field(default=False)
    cuda_version: str | None = Field(default=None)
    rocm_available: bool = Field(default=False)
    vendor: str = Field(default="unknown", description="GPU vendor: nvidia, amd, apple, intel")


def detect_gpu() -> GPUInfo | None:
    """Detect GPU and return GPUInfo, or None if no usable GPU found.

    Tries NVIDIA first, then AMD ROCm, then Apple Silicon unified memory.
    """
    # Try NVIDIA
    gpu = _detect_nvidia_gpu()
    if gpu:
        return gpu

    # Try AMD ROCm
    gpu = _detect_amd_gpu()
    if gpu:
        return gpu

    # Try Apple Silicon
    gpu = _detect_apple_gpu()
    if gpu:
        return gpu

    logger.info("No GPU detected -- will use CPU-only mode")
    return None


def _detect_nvidia_gpu() -> GPUInfo | None:
    """Detect NVIDIA GPU using nvidia-smi."""
    try:
        # Query GPU name and memory
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]

        if len(parts) < 5:
            return None

        name = parts[0]
        vram_total_mb = float(parts[1])
        vram_free_mb = float(parts[2])
        vram_used_mb = float(parts[3])
        driver_version = parts[4]

        # Detect CUDA version
        cuda_version = _detect_cuda_version()

        return GPUInfo(
            name=name,
            vram_total_gb=round(vram_total_mb / 1024, 1),
            vram_free_gb=round(vram_free_mb / 1024, 1),
            vram_used_gb=round(vram_used_mb / 1024, 1),
            driver_version=driver_version,
            cuda_available=True,
            cuda_version=cuda_version,
            vendor="nvidia",
        )

    except (subprocess.SubprocessError, FileNotFoundError, ValueError, IndexError) as e:
        logger.debug("NVIDIA GPU detection failed: %s", e)
        return None


def _detect_cuda_version() -> str | None:
    """Detect installed CUDA version from nvidia-smi or nvcc."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "CUDA Version" in line:
                    # Parse "CUDA Version: 12.4" from the table header
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        version = parts[1].strip().split()[0].strip("|").strip()
                        return version
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fallback to nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    # "Cuda compilation tools, release 12.4, V12.4.99"
                    parts = line.split("release")
                    if len(parts) > 1:
                        return parts[1].strip().split(",")[0].strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


def _detect_amd_gpu() -> GPUInfo | None:
    """Detect AMD GPU using rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return None

        # Parse rocm-smi output for VRAM info
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            return None

        # Get GPU name
        name_result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_name = "AMD GPU"
        if name_result.returncode == 0:
            for line in name_result.stdout.splitlines():
                if "Card" in line or "GPU" in line:
                    # Extract card name
                    parts = line.split(":")
                    if len(parts) > 1:
                        gpu_name = parts[1].strip()
                        break

        # Parse VRAM values from CSV
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    vram_total_mb = float(parts[0].strip()) / (1024 * 1024)
                    vram_used_mb = float(parts[1].strip()) / (1024 * 1024)
                    return GPUInfo(
                        name=gpu_name,
                        vram_total_gb=round(vram_total_mb / 1024, 1),
                        vram_free_gb=round((vram_total_mb - vram_used_mb) / 1024, 1),
                        vram_used_gb=round(vram_used_mb / 1024, 1),
                        rocm_available=True,
                        vendor="amd",
                    )
                except (ValueError, IndexError):
                    continue

    except (subprocess.SubprocessError, FileNotFoundError) as e:
        logger.debug("AMD GPU detection failed: %s", e)

    return None


def _detect_apple_gpu() -> GPUInfo | None:
    """Detect Apple Silicon GPU (unified memory)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    try:
        # Get chip name
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        chip_name = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"

        # Get total memory (unified memory serves as both RAM and VRAM)
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            total_bytes = int(result.stdout.strip())
            total_gb = round(total_bytes / (1024**3), 1)
            # Apple Silicon uses unified memory; estimate ~75% available for GPU
            gpu_available_gb = round(total_gb * 0.75, 1)

            return GPUInfo(
                name=f"{chip_name} (Unified Memory)",
                vram_total_gb=gpu_available_gb,
                vram_free_gb=None,  # Can't easily determine free unified memory
                vendor="apple",
            )

    except (subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        logger.debug("Apple GPU detection failed: %s", e)

    return None
