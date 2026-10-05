"""Reusable Streamlit components for the RadQueue AI dashboard."""

import sys
from pathlib import Path

# Allow `streamlit run dashboard/app.py` from the project root to import `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
