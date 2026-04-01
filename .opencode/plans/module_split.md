# Module Split Plan

## Current State
- `run_models_tui.py`: TUI-based server management (Rich UI, GPU monitoring)
- `run_opencode_models.py`: CLI-based server management (ANSI colors, background/foreground modes)

## Shared Code Identified
1. **Config dataclass** - Server defaults (paths, ports, model paths, parameters)
2. **ServerConfig dataclass** - Individual server configuration
3. **build_server_cmd()** - Command builder for llama-server
4. **Validation functions** - validate_port, validate_threads, require_model, require_executable, validate_ports
5. **Config creators** - create_summary_balanced_cfg, create_summary_fast_cfg, create_qwen35_cfg

## Unique Code by File

### run_models_tui.py
- **TUIApp class** - Main Rich-based TUI with dynamic 2-column layout
- **LogBuffer class** - Thread-safe log buffer with autoscroll
- **GPUStats class** - GPU monitoring via nvtop/psutil

### run_opencode_models.py
- **ServerManager class** - Process management (start, stop, cleanup)
- **dry_run() function** - Preview commands without executing
- **CLI argument parsing** - Positional arguments (mode, ports)
- **Server execution logic** - Background/foreground modes

## Proposed Package Structure

```
llm-runner/
├── llama_manager/              # Core package
│   ├── __init__.py            # Package exports
│   │
│   ├── config.py              # Config & ServerConfig dataclasses
│   │
│   ├── server.py              # build_server_cmd() + validation functions
│   │
│   ├── config_builder.py      # create_*_cfg functions
│   │
│   ├── colors.py              # Color class (Rich version)
│   │
│   ├── log_buffer.py          # LogBuffer class
│   │
│   ├── gpu_stats.py           # GPUStats class  
│   │
│   └── process_manager.py     # ServerManager class
│
├── llama_cli/                  # CLI layer
│   ├── __init__.py
│   │
│   ├── tui_app.py             # TUIApp class (from run_models_tui.py)
│   │
│   └── server_runner.py       # Server execution logic (from run_opencode_models.py)
│       ├── cli_parser.py      # CLI argument parsing
│       ├── dry_run.py         # dry_run function
│       └── runner.py          # Main execution logic
│
└── Entry Points (keeping original names)
    ├── run_models_tui.py      # TUI entry point
    └── run_opencode_models.py # CLI entry point
```

## Module Responsibilities

### llama_manager/
- **config.py**: Configuration constants and dataclasses
- **server.py**: Server command building and validation
- **config_builder.py**: ServerConfig creation helpers
- **colors.py**: Color utilities (Rich-based)
- **log_buffer.py**: Log buffering for TUI
- **gpu_stats.py**: GPU statistics collection
- **process_manager.py**: Process lifecycle management

### llama_cli/
- **tui_app.py**: TUI application logic
- **server_runner.py**: CLI execution logic (dry-run, background/foreground)

### Entry Points
- Keep original filenames for backwards compatibility
- Import and delegate to respective modules
- Maintain original CLI interfaces

## Migration Steps

1. **Create package structure**
   - Create `llama_manager/` directory
   - Create `llama_cli/` directory
   - Create `__init__.py` files

2. **Extract shared code to llama_manager**
   - `config.py`: Move Config & ServerConfig
   - `server.py`: Move build_server_cmd() + validation
   - `config_builder.py`: Move create_*_cfg functions
   - `colors.py`: Convert Color to Rich version
   - `log_buffer.py`: Move LogBuffer class
   - `gpu_stats.py`: Move GPUStats class
   - `process_manager.py`: Move ServerManager class

3. **Extract CLI-specific code to llama_cli**
   - `tui_app.py`: Move TUIApp class, adapt imports
   - `cli_parser.py`: Extract CLI argument parsing
   - `dry_run.py`: Extract dry_run function
   - `runner.py`: Extract server execution logic

4. **Update entry points**
   - `run_models_tui.py`: Import and run TUIApp
   - `run_opencode_models.py`: Import and run CLI logic

5. **Verify functionality**
   - Test TUI mode
   - Test CLI modes (summary-balanced, summary-fast, qwen35, both)
   - Test dry-run mode

## Key Decisions

1. **Color class**: Convert ANSI version to Rich (consistent with TUI)
2. **ServerManager**: Move to llama_manager (shared utility)
3. **Validation**: Keep in server.py (shared with command building)
4. **Config builders**: Separate file for clarity
5. **Entry points**: Keep as thin wrappers for backwards compatibility
