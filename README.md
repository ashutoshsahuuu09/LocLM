# LocLM — Universal Fully Offline Local AI Assistant

<p align="center">
  <strong>Local Models + Local Agents + Local Tools + Local Memory = 100% Local Intelligence</strong>
</p>

<p align="center">
  <a href="#-privacy-first"><img src="https://img.shields.io/badge/Privacy-100%25%20Offline-emerald?style=for-the-badge&logo=shield" alt="100% Offline Privacy" /></a>
  <a href="#-hardware-support"><img src="https://img.shields.io/badge/Hardware-Tier%200%20to%20Tier%205-blue?style=for-the-badge&logo=cpu" alt="Hardware Support" /></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT License" /></a>
</p>

---

**LocLM** is a privacy-first, fully offline, terminal-native AI assistant designed to run entirely on your local machine. It dynamically inspects your system hardware (CPU, GPU, RAM, VRAM), selects appropriate quantized LLMs via local engines like [Ollama](https://ollama.ai), and orchestrates multi-agent workflows—with zero cloud dependencies, zero network data egress, and complete user sovereignty.

---

## 🔒 Privacy First Guarantee

- 🛡️ **100% Offline Operation**: Functions completely disconnected from the Internet after initial setup.
- 🚫 **Zero External Telemetry**: No tracking, metrics, analytics, or background telemetry.
- 🔑 **No Cloud APIs Required**: No reliance on OpenAI, Anthropic, Google, or any remote subscription service.
- 💻 **On-Device Inference**: All model weights execute locally on your CPU/GPU hardware.
- 🗄️ **Local Memory & Workspace Storage**: History, tool state, and context persist solely on your local storage.

---

## 📐 Architecture Overview

LocLM implements a modular **V9 Enterprise Architecture featuring Multi-Directory Workspace Management, Directory Router, Granular Permissions (READ, WRITE, FULL), Project Creation Agent, and Built-in Templates** backed by intelligent hardware profiling and sandboxed local execution tools.

<p align="center">
  <img src="assets/loclm_architecture_visual.jpg" alt="LocLM Architecture Diagram" width="100%" />
</p>

### System Architecture Flow (Mermaid Diagram)

```mermaid
flowchart TD
    subgraph UserInterface["Terminal & Interface"]
        CLI["loclm CLI / Interactive REPL"]
    end

    subgraph WorkspaceLayer["V9 Multi-Directory Workspace Core"]
        WorkspaceManager["Workspace Manager"]
        Registry["Workspace Registry<br/>(~/.loclm/workspaces.json)"]
        DirRouter["Directory Router<br/>(Prompt Matching & Ambiguity Check)"]
        PermissionGuard["Permission Guard<br/>(READ, WRITE, FULL, DENY)"]
        PathGuard["Path Guard<br/>(Path Traversal Prevention)"]

        WorkspaceManager --> Registry & DirRouter & PermissionGuard & PathGuard
    end

    subgraph ProjectLayer["V9 Project Creation Core"]
        ProjectManager["Project Manager"]
        CreatorAgent["ProjectCreationAgent<br/>(Plan ➔ Confirm ➔ Create ➔ Verify)"]
        TemplateReg["Template Registry<br/>(FastAPI, React, Vite, ML, Node, Docker)"]
        Inspector["Project Inspector"]
        Verifier["Project Verifier"]

        ProjectManager --> CreatorAgent & TemplateReg & Inspector & Verifier
    end

    subgraph V8Orchestration["V8/V9 Multi-Agent Swarm Core"]
        V9Orchestrator["V9 Orchestrator Engine"]
        ConsensusEngine["Consensus Manager"]
        SwarmEngine["Swarm Orchestrator"]
        DAGEngine["Workflow DAG Engine"]
        Evaluator["EvaluatorAgent"]

        V9Orchestrator --> ConsensusEngine & SwarmEngine & DAGEngine & Evaluator
    end

    CLI --> WorkspaceManager
    WorkspaceManager --> ProjectManager
    ProjectManager --> V9Orchestrator
```

---

## 🧩 Architectural Components

### 1. ⚙️ Hardware Profiler & Tier Engine (`loclm.hardware`)
LocLM automatically benchmarks your hardware environment on launch:
- **System Memory**: Detects available physical and virtual RAM.
- **Accelerator Detection**: Auto-detects NVIDIA CUDA GPUs, Apple Silicon Metal (unified memory), and AMD ROCm.
- **Hardware Tiers**: Automatically classifies hardware into Tiers (0 to 5) to adjust model parameters (context window size, batching, thread pool concurrency, quantization level).

### 2. 📁 V9 Multi-Directory Workspace Manager (`loclm.workspace`)
- **Workspace Registry (`~/.loclm/workspaces.json`)**: Local, offline storage for approved directory paths and permissions.
- **Granular Permissions**:
  - `READ`: List, view, search, analyze. Edits/creations blocked.
  - `WRITE`: Create, modify, rename files. Destructive ops require confirmation.
  - `FULL`: Full workspace operation rights.
- **Directory Router**: Automatically routes user prompts (e.g., `"Fix auth in CodeV"`) to the corresponding approved workspace. Prompts user if request is ambiguous across multiple projects.
- **Path Traversal Prevention (`PathGuard`)**: Blocks `../` path traversal attempts escaping approved workspace boundaries.

### 3. 🏗️ V9 Project Creation Engine (`loclm.projects`)
- **ProjectCreationAgent**: Natural language requirement parsing, architectural blueprint planning, file tree creation, code generation, and verification.
- **Built-in Templates**: `FastAPI`, `React`, `Vite`, `Python CLI`, `Machine Learning`, `Data Science`, `Node.js`, `Express`, `Full Stack`, `Docker`.
- **Existing File Protection**: Never overwrites files silently (`keep`, `replace`, `backup & replace`).

### 4. 🤖 V8/V9 Multi-Agent System & Swarm Core (`loclm.agents`)
- **V9 Orchestrator Engine**: Coordinates Workspace Management, Directory Routing, Project Creation, Swarm Deliberation, Consensus Plan Synthesis, Workflow DAG Execution, and Output Evaluation.
- **Multi-Agent Swarm Orchestrator (`SwarmOrchestrator`)**: Coordinates multi-round agent interactions across **Leader**, **Worker**, **Critic**, and **Verifier** roles.
- **Consensus Engine (`ConsensusManager`)**: Synthesizes unified high-confidence plans across agent roles.
- **EvaluatorAgent**: Quality scoring on correctness, completeness, safety, and relevance with automatic retry triggers.
- **Specialized Agents**:
  - `CodingAgent`: Handles code analysis, refactoring, patch creation, and syntax validation.
  - `TerminalAgent`: Interprets natural language requests into shell operations with built-in safety boundaries.
  - `KnowledgeAgent`: Synthesizes project structure, documentation, and localized knowledge items.
  - `GeneralAgent`: Handles general dialogue, reasoning, and standard assistant interactions.
  - `RefactorAgent` *(V6)*: AST-guided safe symbol renaming and zero-regression refactoring.
  - `ProjectAgent` *(V6)*: Full project scaffolding and file generation from natural language.

### 5. 🛠️ Sandboxed Tool System (`loclm.tools`)
- **Filesystem**: Safe file reading, modification, tree traversal, and diff generation within workspace boundaries.
- **Terminal Sandbox**: Executes commands with explicit permission enforcement (`READ_ONLY`, `USER_CONFIRM`, `FULL_CONTROL`).
- **Python Sandbox**: Isolated Python runtime execution for mathematical modeling and data processing.
- **Git Integration**: Inspects repository status, log histories, commit patterns, and branch status cleanly.

---

## 🖥️ Hardware Tier System

LocLM matches your hardware configuration to optimal model tiers:

| Tier | Hardware Profile | Recommended Models | Max Context |
| :--- | :--- | :--- | :--- |
| **TIER 0** | CPU + <8GB RAM | `tinyllama`, `qwen2.5:0.5b` | 2,048 tokens |
| **TIER 1** | CPU + 8–16GB RAM | `qwen2.5:1.5b`, `llama3.2:1b` | 4,096 tokens |
| **TIER 2** | GPU 4–8GB VRAM / Apple M-Series (8–16GB) | `qwen2.5:3b`, `qwen2.5:7b-q4` | 8,192 tokens |
| **TIER 3** | GPU 8–16GB VRAM / Apple M-Series (16–32GB) | `qwen2.5:7b`, `mistral:7b`, `deepseek-r1:7b` | 16,384 tokens |
| **TIER 4** | GPU 16–24GB VRAM / Apple M-Series (32–64GB) | `qwen2.5:14b`, `codellama:13b`, `mistral-nemo` | 32,768 tokens |
| **TIER 5** | GPU 24GB+ VRAM / Apple M-Series (64GB+) | `qwen2.5:32b`, `llama3.3:70b-q4` | 65,536+ tokens |

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.12+**
- **[Ollama](https://ollama.ai)** installed and running in the background (`ollama serve`)

### 1. Installation

Clone the repository and install LocLM in editable mode:

```bash
git clone https://github.com/ashutoshsahuuu09/LocLM.git
cd LocLM
pip install -e .
```

### 2. Pull a Local Model

Pull your preferred model via Ollama (e.g., Qwen 2.5):

```bash
ollama pull qwen2.5:3b
```

### 3. Basic Execution

```bash
# Launch interactive REPL mode
loclm

# Run a one-shot task directly from terminal
loclm "Create a Python script that formats JSON logs"

# Inspect hardware tier and status
loclm status

# Run system health diagnostics
loclm doctor
```

---

## 🛠️ CLI Command Reference

| Command | Usage | Description |
| :--- | :--- | :--- |
| `loclm` | `loclm` | Launches the interactive terminal REPL interface |
| `loclm "prompt"` | `loclm "Explain AsyncIO"` | Executes a prompt directly in one-shot mode |
| `loclm chat` | `loclm chat` | Explicit chat mode with session persistence |
| `loclm status` | `loclm status` | Displays detected hardware, VRAM, and active Tier |
| `loclm doctor` | `loclm doctor` | Validates local dependencies (Ollama, Python, SQLite) |
| `loclm models` | `loclm models` | Lists available local models and status |
| `loclm config` | `loclm config` | View and edit user configurations |
| `loclm version` | `loclm version` | Prints current LocLM version details |

---

## 📂 Project Structure

```
LocLM/
├── assets/                       # README Visual Assets & Diagrams
│   ├── loclm_hero_banner.jpg
│   └── loclm_architecture_visual.jpg
├── config/                       # Configuration schemas & default profiles
├── loclm/                        # Core LocLM Python Package
│   ├── agents/                   # Router, Planner, Loop Runner, Specialized Agents
│   │   ├── evaluator_agent.py    # V7: Quality scoring & retry orchestration
│   │   ├── v7_orchestrator.py    # V7: Plugin + Session + Parallel + Eval pipeline
│   │   └── v5_orchestrator.py   # V5: Self-correction orchestrator
│   ├── cli/                      # Rich/Typer Terminal Interface
│   ├── core/                     # Orchestrator & State Management
│   ├── hardware/                 # CPU/GPU Detector & Tier Profiler
│   ├── memory/                   # Offline Memory
│   │   └── session.py            # V7: Session-persistent conversation thread
│   ├── models/                   # Ollama Manager & Model Selector
│   ├── plugins/                  # V7: Runtime Plugin Discovery & Registry
│   │   ├── loader.py             # *_plugin.py file scanner & importer
│   │   └── registry.py          # Tool/agent hot-registration from plugins
│   ├── repo/                     # V6: Multi-repo AST index & symbol search
│   ├── security/                 # Isolation & Network Security Guards
│   ├── testing/                  # V6: Regression test runner
│   └── tools/                    # Sandboxed Tools (FS, Terminal, Python, Git)
├── tests/                        # Comprehensive Pytest Suite
│   ├── test_v7_pipeline.py       # V7: Plugin, Session, Eval, Parallel tests
│   └── test_v6_refactoring.py   # V6: AST refactoring & multi-repo tests
├── pyproject.toml                # Project Build Configuration (v7.0.0)
└── README.md                     # Documentation
```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
