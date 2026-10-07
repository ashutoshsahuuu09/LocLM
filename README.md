# LocLM — Universal Fully Offline Local AI Assistant

> **Local Models + Local Agents + Local Tools + Local Memory = Local Intelligence**

LocLM is a privacy-first, fully offline, terminal-native AI assistant that runs entirely on your computer. It automatically detects your hardware, selects suitable local models, and completes tasks without any cloud AI dependency.

## 🔒 Privacy First

- **100% Offline** after initial setup
- **Zero telemetry** — no data leaves your machine
- **No cloud APIs** — no OpenAI, Anthropic, Google, etc.
- **Local inference** — all models run on your hardware
- **Local memory** — SQLite + FAISS, no cloud databases

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) installed and running

### Installation

```bash
cd LocLM
pip install -e .
```

### Pull a Model

```bash
ollama pull qwen2.5:3b
```

### Run

```bash
# Interactive chat
loclm

# One-shot task
loclm "Explain Python decorators"

# System status
loclm status

# Health check
loclm doctor

# List models
loclm models
```

## 🖥️ Hardware Support

LocLM adapts to your hardware automatically:

| Hardware | Tier | Models |
|----------|------|--------|
| CPU + <8GB RAM | TIER 0 | Tiny models (tinyllama) |
| CPU + 8-16GB RAM | TIER 1 | Small models (qwen2.5:1.5b-3b) |
| GPU 4-8GB VRAM | TIER 2 | Small/Medium quantized (qwen2.5:7b-q4) |
| GPU 8-16GB VRAM | TIER 3 | Medium/Large quantized |
| GPU 16-24GB VRAM | TIER 4 | Large models |
| GPU 24GB+ VRAM | TIER 5 | Advanced models |

## 📐 Architecture

```
CLI → Hardware Detector → Model Manager → Router → Agent → Tools → Memory → Result
```

## 🛠️ CLI Commands

| Command | Description |
|---------|-------------|
| `loclm` | Interactive chat mode |
| `loclm "task"` | One-shot task execution |
| `loclm chat` | Explicit chat mode |
| `loclm status` | System status overview |
| `loclm doctor` | Health diagnostics |
| `loclm models` | Model management |
| `loclm config` | Configuration |
| `loclm version` | Version info |

## 📄 License

MIT
