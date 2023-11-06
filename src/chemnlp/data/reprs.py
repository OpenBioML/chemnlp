import backoff
import deepsmiles
import pubchempy as pcp
import requests
import safe
import selfies
from rdkit import Chem


def smiles_to_selfies(smiles: str) -> str:
    """
    Takes a SMILES and return the selfies encoding.
    """

    return selfies.encoder(smiles)


def smiles_to_deepsmiles(smiles: str) -> str:
    """
    Takes a SMILES and return the DeepSMILES encoding.
    """
    converter = deepsmiles.Converter(rings=True, branches=True)
    return converter.encode(smiles)


def smiles_to_canoncial(smiles: str) -> str:
    """
    Takes a SMILES and return the canoncial SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def smiles_to_inchi(smiles: str) -> str:
    """
    Takes a SMILES and return the InChI.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol)


def smiles_to_safe(smiles: str) -> str:
    """
    Takes a SMILES and return the SAFE.
    """
    return safe.encode(smiles, seed=42, canonical=True, randomize=False)


CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_time=10)
def cactus_request_w_backoff(smiles, rep="iupac_name"):
    url = CACTUS.format(smiles, rep)
    response = requests.get(url, allow_redirects=True, timeout=10)
    response.raise_for_status()
    name = response.text
    if "html" in name:
        return None
    return name


def smiles_to_iupac_name(smiles: str) -> str:
    """Use the chemical name resolver https://cactus.nci.nih.gov/chemical/structure.
    If this does not work, use pubchem.
    """
    try:
        name = cactus_request_w_backoff(smiles, rep="iupac_name")
        if name is None:
            raise Exception
        return name
    except Exception:
        try:
            compound = pcp.get_compounds(smiles, "smiles")
            return compound[0].iupac_name
        except Exception:
            return None
