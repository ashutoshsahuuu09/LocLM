"""LocLM CLI application.

Implements all CLI commands using Typer:
- loclm              -> Interactive chat mode
- loclm ask "task"   -> One-shot task
- loclm chat         -> Explicit interactive chat mode
- loclm status       -> System status overview
- loclm doctor       -> Health diagnostics
- loclm models       -> List installed models
- loclm config       -> Show configuration
- loclm version      -> Version info
"""

from __future__ import annotations

import asyncio
import logging
import sys
import shutil
from typing import Optional

import typer
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from loclm import __version__
from loclm.cli.ui import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
    show_banner,
    show_chat_runtime_info,
    show_doctor_results,
    show_hardware_info,
    show_models_table,
    show_recommendations,
    show_status,
)
from loclm.core.orchestrator import Orchestrator
from loclm.core.state import AppState, get_state
from loclm.security.network import get_network_status

logger = logging.getLogger(__name__)

# Create the Typer app
app = typer.Typer(
    name="loclm",
    help="LocLM -- Universal Fully Offline Local AI Assistant",
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode="rich",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _init_state() -> AppState:
    """Initialize the global application state."""
    state = get_state()
    if not state.is_initialized:
        await state.initialize()
    return state


def _run_async(coro):
    """Run an async coroutine in the event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context -- shouldn't happen with Typer
            return asyncio.ensure_future(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# --------------------------------------------------------------------------
# Main callback (handles bare `loclm` -> interactive chat)
# --------------------------------------------------------------------------

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """LocLM -- Universal Fully Offline Local AI Assistant.

    Run without arguments for interactive chat:
      loclm

    Or use the 'ask' command for one-shot tasks:
      loclm ask "Explain recursion in Python"
    """
    _setup_logging(verbose)

    if ctx.invoked_subcommand is not None:
        # A subcommand was invoked -- let it handle everything
        return

    # No subcommand -> interactive chat
    _run_async(_interactive_chat())


# --------------------------------------------------------------------------
# One-shot task (loclm ask "task")
# --------------------------------------------------------------------------

@app.command()
def ask(
    task: str = typer.Argument(
        ...,
        help="Task to execute (e.g., 'Explain Python decorators')",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Execute a one-shot task and exit.

    Example:
      loclm ask "Explain recursion in Python"
    """
    _setup_logging(verbose)
    _run_async(_oneshot_task(task))


async def _oneshot_task(task: str) -> None:
    """Execute a one-shot task and exit."""
    state = await _init_state()

    if not state.model_manager or not state.model_manager.available_models:
        print_error("No models available. Install one with: ollama pull qwen2.5:3b")
        raise typer.Exit(1)

    orchestrator = Orchestrator(state)
    orchestrator.setup()

    model = state.model_manager.active_model or "unknown"
    console.print(f"[dim]Model: {model}  |  Tier: {state.profile.tier.label}  |  OFFLINE[/]\n")

    if state.config.stream_responses:
        # Stream the response
        full_response: list[str] = []
        async for token in orchestrator.chat_stream(task):
            console.print(token, end="", highlight=False)
            full_response.append(token)
        console.print()  # Final newline
    else:
        response = await orchestrator.chat(task)
        md = Markdown(response)
        console.print(md)

    console.print()
    await state.shutdown()


# --------------------------------------------------------------------------
# V3 Autonomous Coding Agent (loclm code "task")
# --------------------------------------------------------------------------

@app.command()
def code(
    task: str = typer.Argument(
        ...,
        help="Coding task to execute (e.g., 'Search files for ModelSelector and write a test')",
    ),
    max_steps: int = typer.Option(6, "--max-steps", "-s", help="Maximum agent steps"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run the V3 Autonomous Coding Agent on a coding task.

    Example:
      loclm code "List python files and count total lines"
    """
    _setup_logging(verbose)
    _run_async(_run_coding_agent(task, max_steps))


async def _run_coding_agent(task: str, max_steps: int) -> None:
    """Execute Coding Agent loop."""
    state = await _init_state()

    if not state.model_manager or not state.model_manager.available_models:
        print_error("No models available.")
        raise typer.Exit(1)

    from loclm.agents.base import AgentContext
    from loclm.agents.coding import CodingAgent

    agent = CodingAgent(model_manager=state.model_manager, max_steps=max_steps)
    ctx = AgentContext(task=task)

    console.print(f"[bold bright_cyan]LocLM V3 Coding Agent[/]  |  Model: {state.model_manager.active_model}  |  OFFLINE\n")

    response = await agent.run(ctx)

    if response.execution_steps:
        console.print(f"[dim]Executed {len(response.execution_steps)} agent step(s):[/]")
        for step in response.execution_steps:
            if step.tool_call:
                status = "[green]OK[/]" if (step.tool_result and step.tool_result.success) else "[red]FAIL[/]"
                console.print(f"  Step {step.step_number}: Tool [bold yellow]{step.tool_call.tool_name}[/] -> {status}")
        console.print()

    md = Markdown(response.content)
    console.print(md)
    console.print()
    await state.shutdown()


# --------------------------------------------------------------------------
# Interactive chat
# --------------------------------------------------------------------------

@app.command()
def chat(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Start interactive chat mode."""
    _setup_logging(verbose)
    _run_async(_interactive_chat())


async def _interactive_chat() -> None:
    """Run the interactive chat loop."""
    state = await _init_state()

    # Show banner
    show_banner()

    if not state.model_manager or not state.model_manager.available_models:
        print_error("No models available.")
        print_info("Install a model: ollama pull qwen2.5:3b")
        print_info("Then try again: loclm")
        await state.shutdown()
        raise typer.Exit(1)

    # Show runtime info
    show_chat_runtime_info(
        model=state.model_manager.active_model,
        tier=state.profile.tier.label,
        gpu=state.hardware.gpu_name,
    )

    orchestrator = Orchestrator(state)
    orchestrator.setup()

    print_info("Type your message. Use /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = console.input("[loclm.prompt]LocLM > [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n")
            print_info("Goodbye!")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            should_continue = _handle_slash_command(user_input, orchestrator, state)
            if not should_continue:
                break
            continue

        # Process with the model
        console.print()
        if state.config.stream_responses:
            full_response: list[str] = []
            async for token in orchestrator.chat_stream(user_input):
                console.print(token, end="", highlight=False)
                full_response.append(token)
            console.print("\n")
        else:
            response = await orchestrator.chat(user_input)
            md = Markdown(response)
            console.print(md)
            console.print()

    await state.shutdown()


def _handle_slash_command(cmd: str, orchestrator: Orchestrator, state: AppState) -> bool:
    """Handle a slash command in interactive mode.

    Returns:
        True to continue the chat loop, False to exit.
    """
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()

    if command in ("/quit", "/exit", "/q"):
        print_info("Goodbye!")
        return False

    elif command == "/help":
        console.print(
            "\n"
            "[bold]Available commands:[/]\n"
            "  [loclm.key]/help[/]     -- Show this help\n"
            "  [loclm.key]/clear[/]    -- Clear conversation history\n"
            "  [loclm.key]/status[/]   -- Show system status\n"
            "  [loclm.key]/model[/]    -- Show current model info\n"
            "  [loclm.key]/models[/]   -- List installed models\n"
            "  [loclm.key]/quit[/]     -- Exit LocLM\n"
        )

    elif command == "/clear":
        orchestrator.clear_history()
        print_success("Conversation history cleared.")

    elif command == "/status":
        network_status = get_network_status()
        show_status(state.hardware, state.profile, state.model_manager, network_status)

    elif command == "/model":
        if state.model_manager and state.model_manager.active_model:
            console.print(f"[loclm.key]Active model:[/] [loclm.model]{state.model_manager.active_model}[/]")
        else:
            console.print("[dim]No active model[/]")

    elif command == "/models":
        if state.model_manager:
            show_models_table(
                state.model_manager.available_models,
                state.model_manager.active_model,
            )

    else:
        print_warning(f"Unknown command: {command}. Type /help for available commands.")

    console.print()
    return True


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Show LocLM system status."""
    _setup_logging(verbose)

    async def _show_status():
        state = await _init_state()
        network_status = get_network_status()
        show_status(state.hardware, state.profile, state.model_manager, network_status)
        await state.shutdown()

    _run_async(_show_status())


@app.command()
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Run system health diagnostics."""
    _setup_logging(verbose)
    _run_async(_run_doctor())


async def _run_doctor() -> None:
    """Execute all doctor checks."""
    from loclm.hardware.detector import detect_hardware
    from loclm.hardware.profiler import profile_hardware
    from loclm.models.ollama import OllamaRuntime

    results: list[tuple[str, bool, str]] = []

    # 1. Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 12)
    results.append(("Python", py_ok, f"v{py_version}" + ("" if py_ok else " (need 3.12+)")))

    # 2. Hardware detection
    try:
        hardware = detect_hardware()
        results.append(("Hardware Detection", True, f"{hardware.os_name}, {hardware.cpu_name}"))
    except Exception as e:
        hardware = None
        results.append(("Hardware Detection", False, str(e)))

    # 3. RAM
    if hardware:
        ram_ok = hardware.ram_total_gb >= 4
        results.append((
            "RAM",
            ram_ok,
            f"{hardware.ram_total_gb:.0f} GB total, {hardware.ram_available_gb:.1f} GB free",
        ))

    # 4. GPU
    if hardware:
        if hardware.gpu_name:
            results.append(("GPU", True, f"{hardware.gpu_name} ({hardware.gpu_vram_gb:.0f} GB VRAM)"))
        else:
            results.append(("GPU", True, "None detected -- CPU-only mode"))

    # 5. CUDA / Metal
    if hardware:
        if hardware.cuda_available:
            results.append(("CUDA", True, f"v{hardware.cuda_version or 'detected'}"))
        elif hardware.metal_available:
            results.append(("Metal", True, "Available"))
        elif hardware.rocm_available:
            results.append(("ROCm", True, "Available"))
        else:
            results.append(("Acceleration", True, "None -- using CPU"))

    # 6. Storage
    if hardware:
        storage_ok = hardware.storage_free_gb >= 5
        results.append((
            "Storage",
            storage_ok,
            f"{hardware.storage_free_gb:.0f} GB free" + ("" if storage_ok else " (low)"),
        ))

    # 7. Ollama
    ollama_runtime = OllamaRuntime()
    ollama_ok = await ollama_runtime.is_available()
    if ollama_ok:
        results.append(("Ollama", True, "Running on localhost:11434"))
    else:
        results.append(("Ollama", False, "Not running -- start with: ollama serve"))

    # 8. Models
    if ollama_ok:
        models = await ollama_runtime.list_models()
        if models:
            model_names = [m.name for m in models[:3]]
            suffix = f" (+{len(models) - 3} more)" if len(models) > 3 else ""
            results.append(("Local Models", True, ", ".join(model_names) + suffix))
        else:
            results.append(("Local Models", False, "No models installed -- run: ollama pull qwen2.5:3b"))
    else:
        results.append(("Local Models", False, "Cannot check -- Ollama not running"))

    await ollama_runtime.close()

    # 9. Profile tier
    if hardware:
        profile = profile_hardware(hardware)
        results.append(("Performance Tier", True, profile.tier.label))

    # 10. Offline mode
    from loclm.security.network import is_offline_mode
    results.append(("Offline Mode", True, "Enabled" if is_offline_mode() else "Disabled"))

    # Display results
    show_doctor_results(results)


@app.command()
def models(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """List installed models and recommendations."""
    _setup_logging(verbose)

    async def _show_models():
        state = await _init_state()

        if state.model_manager:
            show_models_table(
                state.model_manager.available_models,
                state.model_manager.active_model,
            )

            # Show recommendations
            recs = state.model_manager.get_recommended_models()
            if recs:
                show_recommendations(recs, state.profile.tier.label)

        await state.shutdown()

    _run_async(_show_models())


@app.command(name="config")
def show_config(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Show current configuration."""
    _setup_logging(verbose)

    async def _show_config():
        state = await _init_state()

        from rich.table import Table
        from rich.panel import Panel
        from rich.box import ROUNDED

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="loclm.key", min_width=20)
        table.add_column("Value", style="loclm.value")

        config = state.config
        table.add_row("App Name", config.app_name)
        table.add_row("Version", config.app_version)
        table.add_row("Offline Only", "+ Yes" if config.offline_only else "x No")
        table.add_row("Telemetry", "x Disabled" if not config.telemetry else "+ Enabled")
        table.add_row("Runtime", config.runtime_backend)
        table.add_row("Ollama Host", config.ollama_host)
        table.add_row("Timeout", f"{config.request_timeout}s")
        table.add_row("Streaming", "+ Yes" if config.stream_responses else "x No")
        table.add_row("Max Context", str(config.max_context_messages))
        table.add_row("Max Iterations", str(config.max_iterations))
        table.add_row("Max Retries", str(config.max_retries))

        panel = Panel(
            table,
            title="[loclm.title]LocLM Configuration[/]",
            border_style="bright_cyan",
            box=ROUNDED,
        )
        console.print(panel)

        await state.shutdown()

    _run_async(_show_config())


@app.command(name="tools")
def list_tools(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter tools by category (filesystem, terminal, python, git)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """List all registered V2 local tools and their permission levels."""
    _setup_logging(verbose)

    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ROUNDED
    from loclm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    all_tools = registry.list_tools()

    if category:
        all_tools = [t for t in all_tools if t.category.value == category.lower()]

    table = Table(box=ROUNDED, header_style="bold bright_cyan", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Tool Name", style="bold green", min_width=20)
    table.add_column("Category", style="yellow", min_width=12)
    table.add_column("Permission", style="magenta", min_width=10)
    table.add_column("Description", style="white")

    for idx, tool in enumerate(all_tools, 1):
        perm_style = "green" if tool.permission_level.value == "allow" else "yellow"
        table.add_row(
            str(idx),
            tool.name,
            tool.category.value,
            f"[{perm_style}]{tool.permission_level.value.upper()}[/]",
            tool.description,
        )

    panel = Panel(
        table,
        title=f"[bold bright_cyan]LocLM V2 Local Tools ({len(all_tools)} registered)[/]",
        border_style="bright_blue",
    )
    console.print(panel)


@app.command(name="run-tool")
def run_tool(
    name: str = typer.Argument(..., help="Tool name to execute (e.g. read_file, list_dir, run_python_script, git_status)"),
    args: list[str] = typer.Argument(None, help="Tool arguments in key=value format (e.g. path=README.md, code='print(1)')"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Execute a local tool directly from the terminal.

    Examples:
      loclm run-tool read_file path=README.md
      loclm run-tool list_dir path=.
      loclm run-tool run_python_script code="print(2**10)"
      loclm run-tool git_status
    """
    _setup_logging(verbose)

    from loclm.tools.registry import ToolRegistry

    registry = ToolRegistry()
    kwargs = {}

    if args:
        for arg in args:
            if "=" in arg:
                k, v = arg.split("=", 1)
                kwargs[k.strip()] = v.strip().strip('"').strip("'")
            else:
                kwargs["path"] = arg.strip()

    async def _exec():
        result = await registry.execute_tool(name, **kwargs)
        if result.success:
            print_success(f"Tool '{name}' executed successfully:\n")
            console.print(result.output)
        else:
            print_error(f"Tool '{name}' execution failed: {result.error}")

    _run_async(_exec())


@app.command()
def version() -> None:
    """Show LocLM version."""
    console.print(f"[loclm.title]LocLM[/] [dim]v{__version__}[/]")
