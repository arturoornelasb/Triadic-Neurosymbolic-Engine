"""
Interactive Streamlit Dashboard for the Triadic Neurosymbolic Engine.

Launch with:
    triadic-dashboard
    # or
    python -m neurosym.dashboard
"""
import subprocess
import sys
import os


def main():
    """Launch the Streamlit dashboard."""
    app_path = os.path.join(os.path.dirname(__file__), "_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.headless", "true"],
    )


if __name__ == "__main__":
    main()
