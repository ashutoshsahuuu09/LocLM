"""Rich terminal UI components for LocLM.

Provides styled panels, tables, status displays, and formatted output
using the Rich library.

Note: All output uses ASCII-safe characters for Windows cp1252 compatibility.
"""

from __future__ import annotations

import sys
from typing import Any

from rich.box import ROUNDED, HEAVY, ASCII
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# LocLM color theme
LOCLM_THEME = Theme({
    "loclm.title": "bold bright_cyan",
    "loclm.subtitle": "dim cyan",
    "loclm.success": "bold green",
    "loclm.warning": "bold yellow",
    "loclm.error": "bold red",
    "loclm.info": "bold blue",
    "loclm.muted": "dim white",
    "loclm.accent": "bold magenta",
    "loclm.key": "bold white",
    "loclm.value": "cyan",
    "loclm.prompt": "bold bright_green",
    "loclm.model": "bold yellow",
    "loclm.tier": "bold bright_magenta",
    "loclm.offline": "bold bright_green",
})

# Use ASCII box on Windows legacy terminals to avoid encoding issues
_USE_ASCII = sys.platform == "win32"
_BOX_STYLE = ASCII if _USE_ASCII else ROUNDED

console = Console(theme=LOCLM_THEME)


def show_banner() -> None:
    """Display the LocLM startup banner."""
    banner_text = Text()
    banner_text.append("LocLM", style="bold bright_cyan")
    banner_text.append("\n")
    banner_text.append("Local | Private | Agentic", style="dim cyan")

    panel = Panel(
        banner_text,
        box=_BOX_STYLE,
        border_style="bright_cyan",
        padding=(1, 4),
        title="[bold bright_white][OFFLINE MODE][/]",
        title_align="right",
        subtitle="[dim]v0.1.0[/]",
        subtitle_align="right",
    )
    console.print(panel)


def show_hardware_info(hardware: Any, profile: Any) -> None:
    """Display detected hardware information in a styled panel."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("Key", style="loclm.key", min_width=8)
    table.add_column("Value", style="loclm.value")

    table.add_row("OS", hardware.os_name)
    table.add_row("CPU", hardware.cpu_name)
    table.add_row("Cores", f"{hardware.cpu_cores_physical}P / {hardware.cpu_cores_logical}L")
    table.add_row("RAM", f"{hardware.ram_total_gb:.0f} GB ({hardware.ram_available_gb:.1f} GB free)")

    if hardware.gpu_name:
        table.add_row("GPU", hardware.gpu_name)
        if hardware.gpu_vram_gb:
            vram_text = f"{hardware.gpu_vram_gb:.0f} GB"
            if hardware.gpu_vram_free_gb is not None:
                vram_text += f" ({hardware.gpu_vram_free_gb:.1f} GB free)"
            table.add_row("VRAM", vram_text)
    else:
        table.add_row("GPU", "[dim]None[/]")

    # Acceleration
    accel_parts: list[str] = []
    if hardware.cuda_available:
        cuda_str = "CUDA"
        if hardware.cuda_version:
            cuda_str += f" {hardware.cuda_version}"
        accel_parts.append(cuda_str)
    if hardware.metal_available:
        accel_parts.append("Metal")
    if hardware.rocm_available:
        accel_parts.append("ROCm")
    if not accel_parts:
        accel_parts.append("[dim]None[/]")
    table.add_row("Accel", ", ".join(accel_parts))

    table.add_row("Storage", f"{hardware.storage_free_gb:.0f} GB free / {hardware.storage_total_gb:.0f} GB")

    # Tier
    tier_label = profile.tier.label
    table.add_row("", "")
    table.add_row("[bold]Tier[/]", f"[loclm.tier]{tier_label}[/]")

    panel = Panel(
        table,
        title="[loclm.title]LocLM Hardware[/]",
        border_style="bright_cyan",
        box=_BOX_STYLE,
        padding=(0, 1),
    )
    console.print(panel)


def show_status(
    hardware: Any,
    profile: Any,
    model_manager: Any,
    network_status: dict[str, Any],
) -> None:
    """Display full system status."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        expand=False,
    )
    table.add_column("Key", style="loclm.key", min_width=12)
    table.add_column("Value", style="loclm.value")

    # System
    table.add_row("[bold underline]System[/]", "")
    table.add_row("Mode", "[loclm.offline][OFFLINE][/]" if network_status.get("offline_mode") else "[yellow]ONLINE[/]")
    table.add_row("Runtime", model_manager.runtime.runtime_name if model_manager else "N/A")
    table.add_row("OS", hardware.os_name)
    table.add_row("CPU", hardware.cpu_name)
    table.add_row("RAM", f"{hardware.ram_total_gb:.0f} GB")

    if hardware.gpu_name:
        table.add_row("GPU", hardware.gpu_name)
        if hardware.gpu_vram_gb:
            table.add_row("VRAM", f"{hardware.gpu_vram_gb:.0f} GB")
    else:
        table.add_row("GPU", "[dim]None[/]")

    table.add_row("", "")

    # Models
    table.add_row("[bold underline]Models[/]", "")
    if model_manager:
        mm_status = model_manager.get_status()
        table.add_row("Active", mm_status["active_model"] or "[dim]None[/]")
        table.add_row("Installed", str(mm_status["available_models"]))
        table.add_row("Tier", f"[loclm.tier]{mm_status['tier']}[/]")
        table.add_row("Max Size", f"{mm_status['max_model_size_gb']:.1f} GB")

        # Show cached model selections
        if mm_status.get("model_cache"):
            table.add_row("", "")
            table.add_row("[bold underline]Assignments[/]", "")
            for task, model in mm_status["model_cache"].items():
                table.add_row(f"  {task.title()}", model)

    table.add_row("", "")

    # Network
    table.add_row("[bold underline]Network[/]", "")
    table.add_row(
        "Status",
        "[loclm.success]BLOCKED[/]" if network_status.get("network") == "BLOCKED" else "[yellow]PERMITTED[/]",
    )
    table.add_row("Inference", str(network_status.get("inference", "LOCAL")))
    table.add_row("Memory", str(network_status.get("memory", "LOCAL")))
    table.add_row(
        "Telemetry",
        "[loclm.success]Disabled[/]" if not network_status.get("telemetry") else "[red]Enabled[/]",
    )

    panel = Panel(
        table,
        title="[loclm.title]LocLM Status[/]",
        border_style="bright_cyan",
        box=_BOX_STYLE,
        padding=(0, 1),
    )
    console.print(panel)


def show_doctor_results(results: list[tuple[str, bool, str]]) -> None:
    """Display doctor check results.

    Args:
        results: List of (check_name, passed, detail_message) tuples.
    """
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 1),
    )
    table.add_column("Status", width=3)
    table.add_column("Check", style="bold white", min_width=20)
    table.add_column("Detail", style="dim")

    all_passed = True
    for name, passed, detail in results:
        icon = "[green]+[/]" if passed else "[red]x[/]"
        if not passed:
            all_passed = False
        table.add_row(icon, name, detail)

    panel = Panel(
        table,
        title="[loclm.title]LocLM Doctor[/]",
        border_style="bright_cyan" if all_passed else "yellow",
        box=_BOX_STYLE,
        padding=(0, 1),
    )
    console.print(panel)

    if all_passed:
        console.print("\n[loclm.success]  Status: READY[/]\n")
    else:
        console.print("\n[loclm.warning]  Some checks failed. See details above.[/]\n")


def show_models_table(models: list[Any], active_model: str | None = None) -> None:
    """Display a table of installed models."""
    if not models:
        console.print("[loclm.warning]No models installed.[/]")
        console.print("[dim]Install a model with: ollama pull qwen2.5:3b[/]")
        return

    table = Table(
        box=_BOX_STYLE,
        border_style="bright_cyan",
        title="[loclm.title]Installed Models[/]",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="loclm.model")
    table.add_column("Size", justify="right")
    table.add_column("Family")
    table.add_column("Params")
    table.add_column("Quant")
    table.add_column("Active", justify="center")

    for i, model in enumerate(models, 1):
        is_active = model.name == active_model
        active_mark = "[green]*[/]" if is_active else ""
        name_style = "bold bright_yellow" if is_active else ""

        table.add_row(
            str(i),
            f"[{name_style}]{model.name}[/]" if name_style else model.name,
            f"{model.size_gb:.1f} GB" if model.size_gb > 0 else "-",
            model.family or "-",
            model.parameter_size or "-",
            model.quantization or "-",
            active_mark,
        )

    console.print(table)


def show_recommendations(recommendations: dict[str, str], tier_label: str) -> None:
    """Display model recommendations for the current hardware tier."""
    console.print(f"\n[loclm.info]Recommended models for [loclm.tier]{tier_label}[/] tier:[/]")
    for task, model in recommendations.items():
        console.print(f"  [loclm.key]{task.title():12}[/] -> [loclm.model]{model}[/]")
    console.print()


def show_chat_runtime_info(model: str | None, tier: str, gpu: str | None) -> None:
    """Display runtime info bar during chat."""
    parts: list[str] = [
        "[loclm.offline][OFFLINE][/]",
    ]
    if model:
        parts.append(f"[loclm.model]Model: {model}[/]")
    parts.append(f"[loclm.tier]Tier: {tier}[/]")
    if gpu:
        parts.append(f"[dim]GPU: {gpu}[/]")

    info_text = "  |  ".join(parts)
    console.print(f"[dim]{'-' * 60}[/]")
    console.print(info_text)
    console.print(f"[dim]{'-' * 60}[/]")
    console.print()


def print_response(text: str) -> None:
    """Print a model response with markdown rendering."""
    md = Markdown(text)
    console.print(md)
    console.print()


def print_streaming_token(token: str) -> None:
    """Print a single streaming token without newline."""
    console.print(token, end="", highlight=False)


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[loclm.error]Error:[/] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[loclm.warning]Warning:[/] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[loclm.info]>[/] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[loclm.success]+[/] {message}")
