import os
from glob import glob

import fire

from .model import Dataset


def validate_meta(file):
    """Validate a metadata file."""

    try:
        with open(file, "r") as f:
            _model = Dataset.parse_raw(f.read())  # noqa:  F841
    except Exception as e:
        raise ValueError(f"Error parsing {file}: {e}")


def validate_folder(folder):
    """Validate all metadata files in a folder."""

    files = glob(os.path.join(folder, "**", "meta.yaml"))
    for file in files:
        validate_meta(file)
    return True


if __name__ == "__main__":
    fire.Fire(validate_folder)
