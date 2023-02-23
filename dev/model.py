from pydantic import BaseModel, validator
from pydantic_yaml import YamlStrEnum, YamlModel
from typing import List, Optional, Dict

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
    description: str
    type: IdentifierEnum
    names: List[str]

class ColumnTypes(YamlStrEnum):
    """Column types."""
    
    continuos = "continuos"
    categorical = "categorical"
    ordinal = "ordinal"

class Target(YamlModel):
    """Target information."""
    
    id: str
    description: str
    units: str
    type: ColumnTypes
    names: list[str]

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

class Dataset(YamlModel):
    name : str
    description : str
    targets : Optional[List[Target]]
    identifiers : Optional[List[Identifier]]
    license : str
    num_points : int
    bibtex: List[str]
    templates: Optional[List[Template]]
    fields: Optional[Dict[str, TemplateField]]

    @validator('num_points')
    def num_points_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('num_points must be positive')