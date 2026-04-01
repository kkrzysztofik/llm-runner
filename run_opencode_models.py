#!/usr/bin/env python3
"""
run_opencode_models.py - Manage multiple llama-server instances
"""

import sys

from llama_cli import run_cli

if __name__ == "__main__":
    sys.exit(run_cli(sys.argv))
