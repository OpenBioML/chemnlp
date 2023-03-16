import hashlib
import pathlib
import tarfile
import warnings

import pandas as pd
import requests
import tqdm
import yaml
from thermopyl import Parser

from chemnlp.data_val.model import Dataset


def get_and_transform_data():
    """Downloads the archived version of ThermoML, extracts it and
    loops through the provided JSON-LD files to construct a dataframe.

    """
    # get raw data
    fname = "ThermoML.v2020-09-30.tgz"
    download_path = pathlib.Path(__file__).parent / fname
    remote_data_path = f"https://data.nist.gov/od/ds/mds2-2422/{fname}"
    sha256_checksum = "231161b5e443dc1ae0e5da8429d86a88474cb722016e5b790817bb31c58d7ec2"
    final_csv_path = pathlib.Path(__file__).parent / "thermoml_archive.csv"
    final_expected_csv_checksum = ""

    if not download_path.exists():
        data = requests.get(remote_data_path)
        with open(download_path, "wb") as f:
            for chunk in tqdm.tqdm(
                data.iter_content(chunk_size=8192), desc="Downloading archive"
            ):
                f.write(chunk)

    # check if checksum is correct
    sha256 = hashlib.sha256()
    with open(download_path, "rb") as f:
        for chunk in tqdm.tqdm(iter(lambda: f.read(8192), b""), desc="Checking hash"):
            sha256.update(chunk)

    if received_hash := sha256.hexdigest() != sha256_checksum:
        raise RuntimeError(
            "Downloaded file did not match expected checksum -- "
            "either a new version has been released or something has gone wrong!\n"
            f"Expected: {sha256_checksum}\n"
            f"Received: {received_hash}"
        )

    # Extract tar.gz archive
    with tarfile.open(download_path, "r:*") as tar:
        tar.extractall(pathlib.Path(__file__).parent)

    # Loop through journal DOI folders and scrape files

    if final_csv_path.exists():
        sha256 = hashlib.sha256()
        with open(final_csv_path, "rb") as f:
            for chunk in tqdm.tqdm(
                iter(lambda: f.read(8192), b""), desc="Checking hash"
            ):
                sha256.update(chunk)
        if sha256.hexdigest() != final_expected_csv_checksum:
            warnings.warn(
                "Old CSV file did not match expected checksum, will try to recreate."
            )
        final_csv_path.rename(final_csv_path.with_suffix(".old.csv"))

    root_dois = ("10.1007", "10.1016", "10.1021")

    num_points = 0
    num_failed = 0
    for doi in root_dois:
        for path in tqdm.tqdm(
            (pathlib.Path(__file__).parent / doi).glob("*.xml"),
            desc=f"Looping over files in {doi}",
        ):
            with open(path, "r") as f:
                try:
                    pd.DataFrame(Parser(path).parse()).to_csv(final_csv_path, mode="a")
                    num_points += 1
                except Exception:
                    num_failed += 1

    print(f"Ingested {num_points} with {num_failed} failures.")

    sha256 = hashlib.sha256()
    with open(final_csv_path, "rb") as f:
        for chunk in tqdm.tqdm(iter(lambda: f.read(8192), b""), desc="Checking hash"):
            sha256.update(chunk)

    if csv_hash := sha256.hexdigest() != final_expected_csv_checksum:
        warnings.warn(
            "Final CSV file did not match expected checksum!\n"
            f"Expected: {final_expected_csv_checksum}\n"
            f"Received: {csv_hash}"
        )

    # create metadata
    meta = Dataset(
        **{
            "name": "thermoml_archive",
            "description": "ThermoML is an XML-based IUPAC standard for the storage and exchange of experimental thermophysical and thermochemical property data. The ThermoML archive is a subset of Thermodynamics Research Center (TRC) data holdings corresponding to cooperation between NIST TRC and five journals.",  # noqa
            "identifiers": [
                {
                    "id": "",
                    "type": "inchi",
                },
                {
                    "id": "",
                    "type": "inchikey",
                },
            ],
            "license": "https://www.nist.gov/open/license",
            "links": [
                {
                    "url": "https://doi.org/10.18434/mds2-2422",
                    "description": "data publication",
                },
                {
                    "url": "https://www.nist.gov/publications/towards-improved-fairness-thermoml-archive",
                    "description": "NIST publication description",
                },
                {
                    "url": "https://trc.nist.gov/ThermoML",
                    "description": "Live database hosted at NIST Thermodynamics Research Center",
                },
            ],
            "num_points": num_points,
            "bibtex": [
                "@article{Riccardi2022,title = {Towards improved {{FAIRness}} of the {{ThermoML Archive}}},author = {Riccardi, Demian and Trautt, Zachary and Bazyleva, Ala and Paulechka, Eugene and Diky, Vladimir and Magee, Joseph W. and Kazakov, Andrei F. and Townsend, Scott A. and Muzny, Chris D.},year = {2022},journal = {Journal of Computational Chemistry},volume = {43},number = {12},pages = {879--887},doi = {10.1002/jcc.26842},langid = {english}}",  # noqa
            ],
        }
    )
    with open("meta.yaml", "w") as f:
        yaml.dump(meta.dict(), f, sort_keys=False)


if __name__ == "__main__":
    get_and_transform_data()
