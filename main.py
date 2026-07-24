"""Entry point: launches the Streamlit UI.

Equivalent to running `streamlit run app/ui/streamlit_app.py` directly, but convenient as
`python main.py` per the project's run instructions.
"""
import subprocess
import sys
from pathlib import Path

from app.config import STREAMLIT_SERVER_PORT


def main() -> None:
    app_path = Path(__file__).resolve().parent / "app" / "ui" / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(STREAMLIT_SERVER_PORT)],
        check=True,
    )


if __name__ == "__main__":
    main()
