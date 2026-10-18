"""
Script to move various webapp-related stuff into the docs folder where it's supposed to go.
"""

import pathlib
import shutil

steps = [
    ("web/apps/classification", "docs/demos/classification"),
    ("web/apps/semseg", "docs/demos/semseg"),
]


def main():
    """
    Assumes the build-classification step has run (justfile).
    """
    for src, dst in steps:
        src = pathlib.Path(src)
        dst = pathlib.Path(dst)

        # Create destination directory if it doesn't exist
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing destination if it exists
        if dst.exists():
            shutil.rmtree(dst)

        # Copy the directory
        shutil.copytree(src, dst)


if __name__ == "__main__":
    main()
