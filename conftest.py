# conftest.py — pytest configuration for SignalAI
import sys
import os

# Ensure src is importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
