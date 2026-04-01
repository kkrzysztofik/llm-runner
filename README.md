# TUI for llama-server

A terminal-based user interface for managing multiple llama-server instances with live logs, configuration display, and GPU monitoring.

## Setup

```bash
cd /home/kmk/llm-runner/tui
source .venv/bin/activate
```

## Usage

```bash
# Run both models side-by-side
python run_models_tui.py both

# Run with custom ports
python run_models_tui.py both --port 8080 --port2 8081

# Run single model
python run_models_tui.py summary-balanced --port 8080
python run_models_tui.py qwen35 --port 8081
python run_models_tui.py summary-fast

# Get help
python run_models_tui.py --help
```

## Features

- **2-column layout**: View two models side-by-side (auto-switches to single column on small terminals)
- **Live logs**: Real-time stdout/stderr from each server process
- **GPU stats**: Monitor GPU utilization, memory, temperature, and power draw via nvtop
- **Config display**: Shows port, device, context size, threads, and batch size
- **Auto-scroll**: Logs automatically scroll to show newest output
- **Resize support**: Layout adapts to terminal size changes

## GPU Device Mapping

- **NVIDIA (CUDA)**: GPU 0 (RTX 3090) - used by qwen35-coding
- **Intel (SYCL)**: GPU 1 (Arc B580) - used by summary-balanced, summary-fast

## Exit

Press `Ctrl+C` to gracefully stop all servers and exit.
