from pathlib import Path
import itertools
from typing import Dict, Union

import yaml


def load_config(path: Union[str, Path]):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def _get_all_combinations(d: Dict):
    """Generate all possible hyperparameter combinations"""
    keys, values = d.keys(), d.values()
    values_choices = (
        _get_all_combinations(v) if isinstance(v, dict) else v for v in values
    )
    for comb in itertools.product(*values_choices):
        yield dict(zip(keys, comb))
