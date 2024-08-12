import pytest
import pandas as pd
from chemnlp.data.sampler import TemplateSampler
import numpy as np
import re

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


@pytest.fixture
def sample_config_with_wrapping():
    return {
        'DEFAULT_SIGNIFICANT_DIGITS': 2,
        'multiple_choice_rnd_symbols': ["", ".)", ")"],
        'multiple_choice_benchmarking_templates': False,
        'multiple_choice_benchmarking_format': None,
        'wrap_identifiers': True
    }

# Add these to your existing fixtures or create new ones as needed
@pytest.fixture
def large_sample_df():
    np.random.seed(42)
    return pd.DataFrame({
        'SMILES': [f'C{i}' for i in range(1000)],
        'CYP2D6_Substrate': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
        'LogP': np.random.normal(2, 1, 1000),
        'compound_name': [f'Compound_{i}' for i in range(1000)],
        'split': np.random.choice(['train', 'test', 'valid'], size=1000, p=[0.8, 0.1, 0.1])
    })

@pytest.fixture
def large_sample_meta(sample_meta):
    sample_meta['targets'].append({
        "id": "LogP",
        "type": "continuous",
        "description": "Logarithm of the partition coefficient",
        "names": [{"noun": "LogP value"}, {"noun": "partition coefficient"}],
        "units": "log units",
        "significant_digits": 2
    })
    return sample_meta


def test_basic_identifier_wrapping(sample_df, sample_meta, sample_config_with_wrapping):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config_with_wrapping)
    template = "SMILES: {SMILES#}, Name: {compound_name#}"
    result = sampler.sample(sample_df.iloc[0], template)
    print(result)
    assert "[BEGIN_SMILES]" in result and "[END_SMILES]" in result
    assert "[BEGIN_Other]" in result and "[END_Other]" in result

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
    template = """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {SMILES#} {CYP2D6_Substrate__names__verb}?
Constraint: Even if you are uncertain, you must pick either {%multiple_choice_enum%2%aA1} without using any other words.
Options:
{CYP2D6_Substrate%}
Answer: {%multiple_choice_result}"""
    result = sampler.sample(sample_df.iloc[0], template)
    assert "CC(C)NCC(O)c1ccc(O)c(O)c1" in result
    assert "A or B" in result or "a or b" in result or "1 or 2" in result
    # Check that the answer is one of the options
    answer_letter = result.split("Answer: ")[1].strip()
    # assert that "True" in in the line starting with the answer letter
    assert "True" in [line for line in result.split("\n") if line.startswith(answer_letter) and ('True' in line or 'False' in line)][0]

def test_class_balancing(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    sampler.enable_class_balancing("CYP2D6_Substrate")
    balanced_df = sampler.df
    assert len(balanced_df[balanced_df['CYP2D6_Substrate'] == 0]) == len(balanced_df[balanced_df['CYP2D6_Substrate'] == 1])



def test_class_balancing_large_dataset(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    sampler.enable_class_balancing("CYP2D6_Substrate")
    assert len(sampler.df) < len(large_sample_df)
    assert len(sampler.df[sampler.df['CYP2D6_Substrate'] == 0]) == len(sampler.df[sampler.df['CYP2D6_Substrate'] == 1])


def test_class_balancing_disable(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    sampler.enable_class_balancing("CYP2D6_Substrate")
    assert len(sampler.df) < len(large_sample_df)

    sampler.disable_class_balancing()
    assert len(sampler.df) == len(large_sample_df)
    assert (sampler.df['CYP2D6_Substrate'].value_counts() != sampler.df['CYP2D6_Substrate'].value_counts().iloc[0]).any()

def test_continuous_value_formatting(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    template = "The {LogP__names__noun} of {compound_name#} is {LogP#} {LogP__units}."
    result = sampler.sample(large_sample_df.iloc[0], template)

    assert "LogP value" in result or "partition coefficient" in result
    assert "log units" in result
    assert re.search(r'\d+\.\d{2} log units', result)  # Check if the value is rounded to 2 decimal places

def test_error_handling_invalid_variable(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    template = "This is an {invalid_variable#}."

    with pytest.raises(KeyError):
        sampler.sample(sample_df.iloc[0], template)

def test_multiple_targets_in_template(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    template = "The molecule {compound_name#} with {SMILES__description} {SMILES#} has a {LogP__names__noun} of {LogP#} {LogP__units} and is {CYP2D6_Substrate#not &NULL}a {CYP2D6_Substrate__names__noun}."
    result = sampler.sample(large_sample_df.iloc[0], template)
    print(result)
    assert all(x in result for x in ['Compound_', 'C',  'log units', 'CYP'])
    assert ('is a' in result and 'not a' not in result) or ('is not a' in result and 'is a' not in result)

def test_random_sampling(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    template = "The {compound_name#} has a {LogP__names__noun} of {LogP#}."

    # Sample multiple times without specifying a row
    results = [sampler.sample(None, template) for _ in range(10)]

    # Check if we have at least two different results (high probability)
    assert len(set(results)) > 1


def test_multiple_identifier_types(sample_df, sample_meta, sample_config_with_wrapping):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config_with_wrapping)
    template = "SMILES: {SMILES#}, Name: {compound_name#}"
    result = sampler.sample(sample_df.iloc[0], template)
    assert all(tag in result for tag in ["[BEGIN_SMILES]", "[END_SMILES]",  "[BEGIN_Other]", "[END_Other]"])


def test_wrapping_with_multiple_choice(sample_df, sample_meta, sample_config_with_wrapping):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config_with_wrapping)
    template = """
    Which compound has this SMILES: {SMILES#}?
    {%multiple_choice_enum%2%aA1}
    {compound_name%}
    Answer: {%multiple_choice_result}
    """
    result = sampler.sample(sample_df.iloc[0], template)
    assert "[BEGIN_SMILES]" in result and "[END_SMILES]" in result
    assert "A or B" in result or "a or b" in result or "1 or 2" in result


def test_wrapping_with_continuous_value(large_sample_df, large_sample_meta, sample_config_with_wrapping):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config_with_wrapping)
    template = "SMILES: {SMILES#}, LogP: {LogP#}"
    result = sampler.sample(large_sample_df.iloc[0], template)
    assert "[BEGIN_SMILES]" in result and "[END_SMILES]" in result
    assert re.search(r"LogP: \d+\.\d{2}", result)  # Checks for 2 decimal places
