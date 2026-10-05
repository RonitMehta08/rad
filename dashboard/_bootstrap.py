"""Put the project root on sys.path so pages can import `src` and `dashboard`.

`streamlit run dashboard/app.py` only adds `dashboard/` to the path; every page
imports this module first.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
