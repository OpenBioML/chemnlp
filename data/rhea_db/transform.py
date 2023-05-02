import json
import logging
import os
import subprocess as sp
import tempfile
import time
import urllib
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from bs4 import BeautifulSoup
from rdkit import Chem, RDLogger
from ruamel.yaml import YAML
from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)

# disable nasty rdkit logs
RDLogger.DisableLog("rdApp.*")

BASE_PATH = Path("Downloads")
HTML_PATH = BASE_PATH / "rhea_html"
JSON_PATH = BASE_PATH / "rhea_json"


# ChEBI ID to molecules
# This covers many molecules and avoid extra HTTP requests.
# Seems it still fails sometimes, especially for secondary IDs, fragments, etc

# only contains ChEBI ids with valid InChis.
filename = "https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/chebiId_inchi.tsv"
chebi_inchis = pd.read_csv(filename, sep="\t")
chebi_inchis["smiles"] = [
    Chem.MolToSmiles(Chem.MolFromInchi(inchi, sanitize=False))
    for inchi in tqdm(chebi_inchis.InChI)
]

ID2SMILES = {row["CHEBI_ID"]: row["smiles"] for i, row in chebi_inchis.iterrows()}


def parse_rhea_reactions(
    filename: Union[str, Path] = BASE_PATH / "rhea.rdf"
) -> tuple[pd.DataFrame, dict, dict]:
    """Parse Rhea reactions from an .rdf file downloaded from
    https://ftp.expasy.org/databases/rhea/rdf/rhea.rdf.gz
    Outputs: pd.DataFrame with columns: rhea_id, equation, num_reacts, num_prods
    """
    reacts = {
        "rhea_id": [],
        "equation": [],
    }

    # name, ChEBI_id
    compounds_name2chebi = {}

    with open(filename, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "<rh:equation>" in line:
            # i-1 = label, i-2: id
            id_ = (
                lines[i - 2]
                .split("<rh:accession>")[-1]
                .split("</rh:accession>")[0]
                .split(":")[-1]
            )
            eq = line.split("<rh:equation>")[-1].split("</rh:equation>")[0]
            eq = eq.replace("&gt;", ">").replace("&lt;", "<")
            reacts["rhea_id"].append(id_)
            reacts["equation"].append(eq)

        if "<rh:accession>CHEBI" in line:
            chem_name = lines[i + 1].split("<rh:name>")[1].split("</rh:name>")[0]
            chebi_id = line.split("<rh:accession>CHEBI:")[1].split("</rh:accession>")[0]
            compounds_name2chebi[chem_name] = int(chebi_id)

    compounds_chebi2name = {v: k for k, v in compounds_name2chebi.items()}

    react_ids = pd.DataFrame(reacts)

    react_ids["num_reacts"] = [
        eq.split("=")[0].count(" + ") + 1 for eq in react_ids.equation
    ]
    react_ids["num_prods"] = [
        eq.split("=")[-1].count(" + ") + 1 for eq in react_ids.equation
    ]

    return react_ids, compounds_name2chebi, compounds_chebi2name


def download_multiple(
    urls: Optional[list] = None,
    route: Union[Path, str] = None,
    max_concurrent: int = 5,
) -> Path:
    """Downloads multiple files (form list or file).
    Inputs:
    * urls: list of URLs to download.
    * route: str or Path. Path to store the file.
    * max_concurrent: int. Max number of concurrent downloads in aria2c (default=5, see docs)
    Outputs: Path of dir where to find the downloaded files
    """
    if route is None:
        route = Path(tempfile.gettempdir())

    if isinstance(route, str):
        route = Path(route)

    if not route.exists():
        route.mkdir(parents=True)

    # create temp file
    path = route / "urls_to_download.txt"
    with open(path, "w") as f:
        urls = "\n".join(urls)
        f.write(urls)

    logger.info(f"Downloading in batch: {len(urls)} files")
    if os.system("which aria2c") == 0:
        sp.call(f"aria2c -i {str(path)} -d {route} -j {max_concurrent}", shell=True)
    else:
        logger.warning(
            "For batched downloads, you could get big speedups by installing aria2c"
        )
        sp.call(f"wget -i {str(path)} -P {route}", shell=True)
    logger.info("Downloaded all files")

    return route


@lru_cache
def save_chebiid_smiles(chebi_id: str) -> Chem.Mol:
    """Tries to get a molecule object from a chebi id"""
    smiles = ID2SMILES.get(chebi_id, None)
    if smiles is None:
        with urllib.request.urlopen(
            f"https://www.ebi.ac.uk/chebi/saveStructure.do?sdf=true&chebiId={chebi_id}&imageId=0"
        ) as f:
            molblock = f.read().decode("latin-1")
        molblock = molblock[1:] if molblock.startswith("\n\n") else molblock
        mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
        else:
            logger.error(f"ID: {chebi_id} failed mol processing. Trying scrapping")

            # retrieval from mol failed, try pure scraping
            try:
                with urllib.request.urlopen(
                    f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi_id}"
                ) as f:
                    html = f.read().decode("latin-1").split("\n")
                    for i, line in enumerate(html):
                        if (
                            '<td class="chebiDataHeader" style="width: 150px; height: 15px;">SMILES</td>'
                            in line
                        ):
                            line = (
                                html[i + 1]
                                .lstrip(" ")
                                .lstrip("\t")
                                .lstrip(" ")
                                .lstrip("\t")
                            )
                            soup = BeautifulSoup(line, "html.parser")
                            smiles = soup.text

            except Exception as e:
                logger.error(
                    f"Error while scrapping: {e}. Setting smiles to empty for CHEBI:{chebi_id}"
                )
                smiles = ""

    return smiles


def parse_rhea_id(
    id_: int, filepath: Optional[Union[Path, str]] = None, encoding: str = "latin-1"
) -> tuple[str, dict[int, str]]:
    """Loads a reaction page from Rhea and parses its reactants and products.
    # !wget https://www.rhea-db.org/rhea/57036 -O downloads/57036.txt
    Inputs:
    * id_: int. rhea identifier.
    * filepath: Path or str. Path to the Rhea HTML for a given id.
    Outputs: equation, {chebi_id: mol_name}
    """
    if filepath is not None:
        with open(filepath, "r", encoding=encoding) as f:
            html = f.read()

        # bad encoding sometimes
        if 'href="http://www.ebi.ac.uk/' not in html:
            logger.warning(f"Loaded content from filepath is useless. {filepath}")
            filepath = None

    if filepath is None:
        try:
            with urllib.request.urlopen(f"https://www.rhea-db.org/rhea/{id_}") as f:
                html = f.read().decode(encoding)
        except urllib.error.HTTPError as e:
            logger.error(f"Rhea id failed url access: {id_} with exception {e}")
            return "", {}

    lines = html.split("\n")
    # remove trashy line starts and empty ones
    # lines = list(map(lambda x: x.lstrip(" ").lstrip("\t").lstrip(" ").lstrip("\t"), lines))
    lines = list(filter(len, lines))

    chebi2mol = {}
    eq_filled = False
    for line in lines:
        # get mol names
        if 'onclick="window.attachToolTip(this)" class="molName" data-molid=' in line:
            soup = BeautifulSoup(line, "html.parser")
            mol_name = soup.a.text

        # get chebi id
        if 'href="http://www.ebi.ac.uk/' in line:
            chebi_id = line.split("CHEBI:")[-1].split("<")[0]
            chebi2mol[int(chebi_id)] = {"mol_name": mol_name}

        # get equation string
        if '<textarea readonly style="transform:scale(0,0)" id="equationtext">' in line:
            if not eq_filled:
                soup = BeautifulSoup(line, "html.parser")
                eq = soup.text.lstrip(" ").rstip(" ")
                eq_filled = True

    return eq, chebi2mol


def parse_rhea_id_with_smiles(
    rhea_id: int, filepath: Optional[Union[Path, str]] = None, **kwargs
) -> tuple[int, str, dict[int, tuple[str, str]]]:
    """Parses a full Rhea reaction to get the Rhea ID, the reaction
    equation and compounds.
    Inputs:
    * rhea_id: int. id for a reaction
    * filepath: Path or str. Path to the Rhea HTML for a given id.
    Outputs: rhea_id, reaction_equation, {chebi_id: (cpd_name,smiles)}
    """
    try:
        eq_, cpds = parse_rhea_id(rhea_id, filepath=filepath, **kwargs)
    except Exception as e:
        raise ValueError(f"ID: {rhea_id} failed scrapping: {e}")

    new_cpds = {}
    for id_, cpd in cpds.items():
        # id for a photon, electron
        if id_ in {30212, 10545}:
            smiles = ""
        else:
            smiles = save_chebiid_smiles(id_)
        new_cpds[id_] = (cpd, smiles)
    return rhea_id, eq_, new_cpds


# # single test
# id_ = 29283
# pre_react = parse_rhea_id(id_=id_)
# full_react = parse_rhea_id_with_smiles(rhea_id=id_)


merged, compound_name2chebi, compound_chebi2name = parse_rhea_reactions()

# Prepare full download

all_ids = merged.rhea_id.values.astype(int).tolist()  # [:100]
all_urls = [f"https://www.rhea-db.org/rhea/{id_}" for id_ in all_ids]
download_multiple(all_urls, route=HTML_PATH, max_concurrent=32)


logger.info(f"Prev length: {len(all_ids)}")
all_ids = list(filter(lambda x: Path(f"{HTML_PATH}/{x}").is_file(), all_ids))
logger.info(f"Post length: {len(all_ids)}")


glob_res = []
step = 5000
total_tac = time.time()
for i in range(0, len(all_ids), step):
    tac = time.time()
    results = [
        parse_rhea_id_with_smiles(id_, filepath=Path(f"Downloads/rhea_html/{id_}"))
        for id_ in tqdm(all_ids[i : i + step])
    ]
    results = [{"rhea_id": x[0], "equation": x[1], "compounds": x[2]} for x in results]
    glob_res.extend(results)
    tic = time.time()
    tot_time = tic - total_tac
    logger.info(
        f"{i} took: {tic - tac:.3f}s for {step} queries. So far: {tot_time:.3f}s: {len(all_ids) - i} remaining"
    )


# Save to json
with open(JSON_PATH / "parsed_rhea.json", "w") as f:
    for item in glob_res:
        json.dump(item, f)
        f.write("\n")


# Verify number of datapoints
yaml = YAML(typ="safe").load(open("meta.yaml").read())
assert yaml["num_points"] == len(
    glob_res
), f"Number of points {len(glob_res)} does not match {yaml['num_points']}"
logger.info("SUCCESS PARSING RHEA DB DATASET")
