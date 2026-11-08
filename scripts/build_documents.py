"""Compatibility entry point for the IEEE conference document builder."""
from pathlib import Path
from runpy import run_path

run_path(str(Path(__file__).with_name('build_ieee.py')), run_name='__main__')
