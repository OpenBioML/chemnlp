from typing import Dict, List, Optional

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


class Identifier(YamlModel):
    """Identifier information."""

    id: str
    description: Optional[str]
    type: IdentifierEnum
    names: Optional[List[str]]

    """A URI or multiple (consitent ) URIs for the field.

    Ideally this would be a link to an entry in an ontrology or controlled
    vocabulary that can also provide a canonical description for the field.
    """
    uri: Optional[List[str]]

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


class Target(YamlModel):
    """Target information."""

    id: str
    description: str
    units: str
    type: ColumnTypes
    names: List[str]


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

    @validator("num_points")
    def num_points_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("num_points must be positive")
