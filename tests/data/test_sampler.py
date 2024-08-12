import pytest
import pandas as pd
from chemnlp.data.sampler import TemplateSampler

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'SMILES': ['CC(C)NCC(O)c1ccc(O)c(O)c1', 'CC1=C(C(=O)NC2=C1C=CC=C2)C3=CC=CC=C3'],
        'CYP2D6_Substrate': [1, 0],
        'compound_name': ['Isoproterenol', 'Phenytoin'],
        'split': ['train', 'test']
    })

@pytest.fixture
def sample_meta():
    return {
        "identifiers": [
            {"id": "SMILES", "type": "SMILES", "description": "SMILES"},
            {"id": "compound_name", "type": "Other", "description": "drug name",
             "names": [{"noun": "compound name"}, {"noun": "drug name"}, {"noun": "generic drug name"}]}
        ],
        "targets": [
            {
                "id": "CYP2D6_Substrate",
                "type": "boolean",
                "description": "drugs that are metabolized by the CYP P450 2D6 (1) or not (0)",
                "names": [
                    {"noun": "CYP P450 2D6 substrate"},
                    {"noun": "CYP2D6 substrate"},
                    {"noun": "substrate for CYP2D6"},
                    {"noun": "substrate for CYP P450 2D6"},
                    {"verb": "metabolized by CYP2D6"},
                    {"verb": "metabolized by CYP P450 2D6"}
                ]
            }
        ]
    }

@pytest.fixture
def sample_config():
    return {
        'DEFAULT_SIGNIFICANT_DIGITS': 2,
        'multiple_choice_rnd_symbols': ["", ".)", ")"],
        'multiple_choice_benchmarking_templates': False,
        'multiple_choice_benchmarking_format': None
    }

def test_get_target_from_row(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    assert sampler._get_target_from_row(sample_df.iloc[0], "SMILES#") == "CC(C)NCC(O)c1ccc(O)c(O)c1"
    assert sampler._get_target_from_row(sample_df.iloc[0], "CYP2D6_Substrate#") == "1"
    assert sampler._get_target_from_row(sample_df.iloc[0], "compound_name#") == "Isoproterenol"

def test_get_target_from_string(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    assert sampler._get_target_from_string("CYP2D6_Substrate__names__noun")() in [
        "CYP P450 2D6 substrate", "CYP2D6 substrate", "substrate for CYP2D6", "substrate for CYP P450 2D6"
    ]
    assert sampler._get_target_from_string("CYP2D6_Substrate__names__verb")() in [
        "metabolized by CYP2D6", "metabolized by CYP P450 2D6"
    ]

def test_sample_with_template(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    template = "The molecule with the {SMILES__description} {SMILES#} is {CYP2D6_Substrate#not &NULL}a {CYP2D6_Substrate__names__noun}."
    result = sampler.sample(sample_df.iloc[0], template)
    assert "CC(C)NCC(O)c1ccc(O)c(O)c1" in result
    assert "is a" in result
    assert "CYP P450 2D6 substrate" in result or "CYP2D6 substrate" in result or "substrate for CYP2D6" in result or "substrate for CYP P450 2D6" in result

def test_multiple_choice_template(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    template = """
    Task: Please answer the multiple choice question.
    Question: Is the molecule with the {SMILES__description} {SMILES#} {CYP2D6_Substrate__names__verb}?
    Constraint: Even if you are uncertain, you must pick either {%multiple_choice_enum%2%aA1} without using any other words.
    Options:
    {CYP2D6_Substrate%}
    Answer: {%multiple_choice_result}
    """
    result = sampler.sample(sample_df.iloc[0], template)
    assert "CC(C)NCC(O)c1ccc(O)c(O)c1" in result
    assert "A or B" in result or "a or b" in result or "1 or 2" in result
    assert result.strip().endswith("A") or result.strip().endswith("a") or result.strip().endswith("1")

def test_class_balancing(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    sampler.enable_class_balancing("CYP2D6_Substrate")
    balanced_df = sampler.df
    assert len(balanced_df[balanced_df['CYP2D6_Substrate'] == 0]) == len(balanced_df[balanced_df['CYP2D6_Substrate'] == 1])
