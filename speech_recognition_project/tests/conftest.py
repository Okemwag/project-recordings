"""
conftest.py — Shared pytest fixtures for the Kiswahili ASR test suite.
"""

import sys
import os

# Ensure the project root is on the Python path so `src` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
