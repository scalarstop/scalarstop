"""Utilities for filesystem access."""
import shutil


def rmtree(path: str) -> None:
    """Remove a filesystem tree. Ignore failures if the tree doesn't exist."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
