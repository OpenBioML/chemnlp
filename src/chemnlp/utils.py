import hashlib
import itertools
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import psutil
import requests
import yaml
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

M_USED = "Memory Used"
M_UTIL = "Memory Utilisation"


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


def extract_tarball(url, output_dir, md5: Optional[str] = None):
    # Download the tarball from the URL
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download tarball from {url}")

    if md5 is not None:
        # check the md5 checksum
        assert compute_md5_of_response(response) == md5

    # Extract the contents of the tarball to the output directory
    with tarfile.open(fileobj=BytesIO(response.content), mode="r|gz") as tar:
        tar.extractall(output_dir)


def xyz_to_mol(xyzfile: Union[Path, str], charge: int = 0, remove_h: bool = False):
    """Read an xyz file into an RDKit molecule object.

    Args:
        xyzfile (Path | str): Filepath
        charge (int, optional): Charge of the molecule. Defaults to 0.
        remove_h (bool, optional): If True, removes hydrogens. Defaults to False.

    Returns:
        Mol: Rdkit molecule object
    """
    raw_mol = Chem.MolFromXYZFile(xyzfile)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    if remove_h:
        mol = Chem.RemoveAllHs(mol)
    return mol


def compute_md5_of_response(response) -> str:
    md5_hash = hashlib.md5()
    for block in response.iter_content(8192):
        md5_hash.update(block)

    return md5_hash.hexdigest()


def add_average_metrics(
    metrics: Dict[str, float], metric_names: List[str], device: str
) -> Dict:
    """Add in averaged metrics for the device"""
    for metric in metric_names:
        all_metric_vals = [val for k, val in metrics.items() if metric in k]
        metrics[f"{device} Avg. {metric}"] = sum(all_metric_vals) / len(all_metric_vals)
    return metrics


def collect_cpu_memory() -> Dict:
    return {f"CPU {M_USED}": psutil.virtual_memory().percent}


def collect_gpu_memory() -> Dict:
    import nvidia_smi

    nvidia_smi.nvmlInit()
    gpu_metrics = {}
    for _id in range(nvidia_smi.nvmlDeviceGetCount()):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(_id)
        mem_usage = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_metrics[f"GPU {_id} {M_USED}"] = int(
            100 * float(mem_usage.used) / float(mem_usage.total)
        )
    nvidia_smi.nvmlShutdown()
    add_average_metrics(gpu_metrics, [M_USED], "GPU")
    return gpu_metrics


def get_local_ip_address() -> str:
    import socket

    hostname = socket.getfqdn()
    return socket.gethostbyname_ex(hostname)[0]


def load_and_split(
    source: str, test_size: float, shuffle: bool = False
) -> datasets.arrow_dataset.Dataset:
    """Loads and splits a HF dataset into train/test"""
    dataset = datasets.load_from_disk(source)
    return dataset.train_test_split(test_size=test_size, shuffle=shuffle)
