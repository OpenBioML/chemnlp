from typing import Dict, List, Optional

import pubchempy as pcp
import requests
from pydantic import root_validator, validator
from pydantic_yaml import YamlModel, YamlStrEnum


class IdentifierEnum(YamlStrEnum):
    """Identifier types."""

    smiles = "SMILES"
    selfies = "SELFIES"
    iupac = "IUPAC"
    inchi = "InChI"
    inchikey = "InChIKey"
    other = "Other"


class SplitEnum(YamlStrEnum):
    """Split types."""

    train = "train"
    valid = "valid"
    test = "test"


class Identifier(YamlModel):
    """Identifier information."""

    id: str

    """A description of the field"""
    description: Optional[str]
    type: IdentifierEnum
    names: Optional[List[str]]

    @root_validator
    def if_optional_has_names(cls, values):
        if (values.get("names") is None) and (
            values.get("type") == IdentifierEnum.other
        ):
            raise ValueError('names must be provided if type is "other"')
        if (values.get("description") is None) and (
            values.get("type") == IdentifierEnum.other
        ):
            raise ValueError('names must be provided if type is "other"')

        return values


class ColumnTypes(YamlStrEnum):
    """Column types."""

    continuous = "continuous"
    categorical = "categorical"
    ordinal = "ordinal"
    boolean = "boolean"


class Target(YamlModel):
    """Target information."""

    id: str

    """A english description of the field"""
    description: str

    """The units of the field. None if unitless."""
    units: Optional[str]

    """The type of the field. Can be one of `continuous`, `categorical`, `ordinal`, `boolean`."""
    type: ColumnTypes

    """A list of names describing the field.

    Note that this will be used in building the prompts. Some example for prompts:

    - Boolean variables

        - `Is <name> <identifier>?`
        - ```
        What molecules in the list are <name>?
        - <identifier_1>
        - <identifier_2>
        - <identifier_3>
        ```


    - Continuous variables

        - `What is <name> of <identifier>?`
        - ```
        What is the molecule with largest <name> in the following list?
        - <identifier_1>
        - <identifier_2>
        - <identifier_3>
        ```
    """
    names: List[str]

    """A URI or multiple (consitent ) URIs for the field.

    Ideally this would be a link to an entry in an ontrology or controlled
    vocabulary that can also provide a canonical description for the field.
    """
    uris: Optional[List[str]]

    """A PubChem assay IDs or multiple (consistent) PubChem assay IDs.

    Make sure that the first assay ID is the primary assay ID.
    """
    pubchem_aids: Optional[List[int]]

    @validator("uris")
    def uris_resolves(cls, values):
        if values is not None:
            for uri in values:
                # perform a request to the URI and check if it resolves
                response = requests.get(uri)
                if response.status_code == 403:
                    print(
                        f"URI {uri} does not resolve (403) since forbidden, please check manually"
                    )
                elif response.status_code != 200:
                    raise ValueError(f"URI {uri} does not resolve")

    @validator("pubchem_aids")
    def pubchem_assay_ids_resolve(cls, values):
        if values is not None:
            for aid in values:
                assays = pcp.get_assays(aid)
                if len(assays) == 0:
                    raise ValueError(f"PubChem assay ID {aid} does not resolve")


class Template(YamlModel):
    prompt: str
    completion: str


class TemplateFieldValue(YamlModel):
    """Template field information."""

    name: str
    column: Optional[str]
    text: Optional[str]


class TemplateField(YamlModel):
    values: List[TemplateFieldValue]


class Link(YamlModel):
    """Link information."""

    url: str
    description: str


class Benchmark(YamlModel):
    """Benchmark information."""

    """The name of the benchmark, e.g. MoleculeNet."""
    name: str

    """The link to the benchmark."""
    link: str

    """The name of the column in the dataset that indicates the fold of the data point."""
    split_column: SplitEnum


class Dataset(YamlModel):
    name: str
    description: str
    targets: Optional[List[Target]]
    identifiers: Optional[List[Identifier]]
    license: str
    num_points: int
    bibtex: List[str]
    templates: Optional[List[Template]]
    fields: Optional[Dict[str, TemplateField]]
    links: List[Link]

    benchmarks: Optional[Benchmark]

    @validator("num_points")
    def num_points_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("num_points must be positive")

    @validator("links")
    def links_must_resolve(cls, v):
        if v is not None:
            for link in v:
                response = requests.get(link.url)
                if response.status_code == 403:
                    print(
                        f"Link {link.url} does not resolve (403) since forbidden, please check manually"
                    )
                elif response.status_code != 200:
                    if not (("acs" in response.text) or ("sage" in response.text)):
                        raise ValueError(f"Link {link.url} does not resolve")
