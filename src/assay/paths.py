"""Where assay keeps its data: the database and the trained model.

This is the operating system's per-user data directory (on Linux, ~/.local/share/assay; elsewhere, whatever
that system uses), so the data survives upgrades and reinstalls of the code. ASSAY_HOME overrides it, which
the tests use. Every other module asks here.
"""
import os
from pathlib import Path

from platformdirs import user_data_dir


def data_dir():
    return Path(os.environ.get("ASSAY_HOME") or user_data_dir("assay"))


def db_path():
    return data_dir() / "assay.db"


def model_path():
    return data_dir() / "model" / "classifier.pkl"
