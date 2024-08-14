import pytest
import pandas as pd
from chemnlp.data.sampler import TemplateSampler
import numpy as np
import re


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "SMILES": [
                "CC(C)NCC(O)c1ccc(O)c(O)c1",
                "CC1=C(C(=O)NC2=C1C=CC=C2)C3=CC=CC=C3",
            ],
            "CYP2D6_Substrate": [1, 0],
            "compound_name": ["Isoproterenol", "Phenytoin"],
            "split": ["train", "test"],
        }
    )


@pytest.fixture
def sample_multiple_identifier_df():
    return pd.DataFrame(
        {
            "SMILES": [
                "CC(C)NCC(O)c1ccc(O)c(O)c1",
                "CC1=C(C(=O)NC2=C1C=CC=C2)C3=CC=CC=C3",
            ],
            "selfies": [
                "[C][C][Branch1][C][C][N][C][C][Branch1][O][C][=C][C][=C][C][Branch1][O][C][=C][Branch1][O][C][=C]",
                "[C][C]1[=C][Branch1][C][Branch2][=O][N][C]2[=C]1[C][=C][C][=C][C]2[C]3[=C][C][=C][C][=C][C]3",
            ],
            "inchi": [
                "InChI=1S/C11H17NO3/c1-8(2)12-6-9(13)10-4-3-5-11(14)7-10/h3-5,7-9,12-14H,6H2,1-2H3",
                "InChI=1S/C15H11NO/c17-14-11-9-5-1-3-7-13(9)16-15(14)10-6-2-4-8-12(10)11/h1-8,16H",
            ],
            "compound_name": ["Isoproterenol", "Phenytoin"],
            "LogP": [0.08, 2.47],
            "is_active": [True, False],
            "split": ["train", "test"],
        }
    )


@pytest.fixture
def sample_meta():
    return {
        "identifiers": [
            {"id": "SMILES", "type": "SMILES", "description": "SMILES"},
            {
                "id": "compound_name",
                "type": "Other",
                "description": "drug name",
                "names": [
                    {"noun": "compound name"},
                    {"noun": "drug name"},
                    {"noun": "generic drug name"},
                ],
            },
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
                    {"verb": "metabolized by CYP P450 2D6"},
                ],
            }
        ],
    }


@pytest.fixture
def sample_multiple_identifier_meta():
    return {
        "targets": [
            {
                "id": "LogP",
                "type": "continuous",
                "description": "Logarithm of partition coefficient",
            },
            {
                "id": "is_active",
                "type": "categorical",
                "description": "Activity status",
            },
        ],
        "identifiers": [
            {"id": "SMILES", "type": "SMILES", "description": "SMILES notation"},
            {"id": "compound_name", "type": "Other", "description": "Compound name"},
        ],
        "templates": [
            "The molecule with SMILES {SMILES#} has a LogP of {LogP#}.",
            "The compound {compound_name#} is {is_active#active&inactive}.",
        ],
    }


@pytest.fixture
def sample_config():
    return {
        "DEFAULT_SIGNIFICANT_DIGITS": 2,
        "multiple_choice_rnd_symbols": ["", ".)", ")"],
        "multiple_choice_benchmarking_templates": False,
        "multiple_choice_benchmarking_format": None,
        "excluded_from_wrapping": ["Other"],
    }


@pytest.fixture
def sample_config_with_wrapping():
    return {
        "DEFAULT_SIGNIFICANT_DIGITS": 2,
        "multiple_choice_rnd_symbols": ["", ".)", ")"],
        "multiple_choice_benchmarking_templates": False,
        "multiple_choice_benchmarking_format": None,
        "wrap_identifiers": True,
        "excluded_from_wrapping": ["Other"],
    }


# Add these to your existing fixtures or create new ones as needed
@pytest.fixture
def large_sample_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "SMILES": [f"C{i}" for i in range(1000)],
            "CYP2D6_Substrate": np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
            "LogP": np.random.normal(2, 1, 1000),
            "compound_name": [f"Compound_{i}" for i in range(1000)],
            "split": np.random.choice(
                ["train", "test", "valid"], size=1000, p=[0.8, 0.1, 0.1]
            ),
        }
    )


@pytest.fixture
def sample_polymer_df():
    return pd.DataFrame(
        {
            "PSMILES": ["*CC(*)C", "*CC(C)C*", "*C(CC)CCC*"],
            "compound_name": [
                "Poly(propylene)",
                "Poly(isobutylene)",
                "Poly(pentylene)",
            ],
            "Tg_exp": [273.15, 200.0, 250.0],
            "Tg_calc": [275.0, 205.0, 245.0],
            "rho_300K_calc": [0.90, 0.92, 0.88],
            "split": ["train", "test", "validation"],
        }
    )


@pytest.fixture
def sample_polymer_meta():
    return {
        "identifiers": [
            {
                "id": "PSMILES",
                "type": "PSMILES",
                "description": "PSMILES representation",
            },
            {
                "id": "compound_name",
                "type": "Other",
                "description": "polymer name",
                "names": [{"noun": "compound name"}, {"noun": "polymer name"}],
            },
        ],
        "targets": [
            {
                "id": "Tg_exp",
                "type": "continuous",
                "description": "Experimental glass transition temperature",
                "units": "K",
                "names": [{"noun": "experimental glass transition temperature"}],
            },
            {
                "id": "Tg_calc",
                "type": "continuous",
                "description": "Computed glass transition temperature",
                "units": "K",
                "names": [{"noun": "computed glass transition temperature"}],
            },
            {
                "id": "rho_300K_calc",
                "type": "continuous",
                "description": "Computed density at 300K",
                "units": "g/cm³",
                "names": [{"noun": "computed density at 300 K"}],
            },
        ],
    }


@pytest.fixture
def sample_polymer_config():
    return {
        "DEFAULT_SIGNIFICANT_DIGITS": 2,
        "multiple_choice_rnd_symbols": ["", ".)", ")"],
        "multiple_choice_benchmarking_templates": False,
        "multiple_choice_benchmarking_format": None,
    }


@pytest.fixture
def large_sample_meta(sample_meta):
    sample_meta["targets"].append(
        {
            "id": "LogP",
            "type": "continuous",
            "description": "Logarithm of the partition coefficient",
            "names": [{"noun": "LogP value"}, {"noun": "partition coefficient"}],
            "units": "log units",
            "significant_digits": 2,
        }
    )
    return sample_meta


def test_basic_identifier_wrapping(sample_df, sample_meta, sample_config_with_wrapping):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config_with_wrapping)
    template = "SMILES: {SMILES#}, Name: {compound_name#}"
    result = sampler.sample(sample_df.iloc[0], template)
    assert "[BEGIN_SMILES]" in result["text"] and "[END_SMILES]" in result["text"]
    assert "[BEGIN_Other]" in result["text"] and "[END_Other]" in result["text"]


def test_get_target_from_row(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    assert (
        sampler._get_target_from_row(sample_df.iloc[0], "SMILES#")
        == "CC(C)NCC(O)c1ccc(O)c(O)c1"
    )
    assert sampler._get_target_from_row(sample_df.iloc[0], "CYP2D6_Substrate#") == "1"
    assert (
        sampler._get_target_from_row(sample_df.iloc[0], "compound_name#")
        == "Isoproterenol"
    )


def test_get_target_from_string(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    assert sampler._get_target_from_string("CYP2D6_Substrate__names__noun")() in [
        "CYP P450 2D6 substrate",
        "CYP2D6 substrate",
        "substrate for CYP2D6",
        "substrate for CYP P450 2D6",
    ]
    assert sampler._get_target_from_string("CYP2D6_Substrate__names__verb")() in [
        "metabolized by CYP2D6",
        "metabolized by CYP P450 2D6",
    ]


def test_sample_with_template(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    template = "The molecule with the {SMILES__description} {SMILES#} is {CYP2D6_Substrate#not &NULL}a {CYP2D6_Substrate__names__noun}."
    result = sampler.sample(sample_df.iloc[0], template)
    assert "CC(C)NCC(O)c1ccc(O)c(O)c1" in result["text"]
    assert "is a" in result["text"]
    assert any(
        substrate in result["text"]
        for substrate in [
            "CYP P450 2D6 substrate",
            "CYP2D6 substrate",
            "substrate for CYP2D6",
            "substrate for CYP P450 2D6",
        ]
    )


def test_multiple_choice_template(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    template = """Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {SMILES#} {CYP2D6_Substrate__names__verb}?
Constraint: Even if you are uncertain, you must pick either {%multiple_choice_enum%2%aA1} without using any other words.
Options:
{CYP2D6_Substrate%}
Answer: {%multiple_choice_result}"""
    result = sampler.sample(sample_df.iloc[0], template)
    assert "CC(C)NCC(O)c1ccc(O)c(O)c1" in result["text"]
    assert any(option in result["text"] for option in ["A or B", "a or b", "1 or 2"])


def test_class_balancing(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    sampler.enable_class_balancing("CYP2D6_Substrate")
    balanced_df = sampler.df
    assert len(balanced_df[balanced_df["CYP2D6_Substrate"] == 0]) == len(
        balanced_df[balanced_df["CYP2D6_Substrate"] == 1]
    )


def test_class_balancing_large_dataset(
    large_sample_df, large_sample_meta, sample_config
):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    sampler.enable_class_balancing("CYP2D6_Substrate")
    assert len(sampler.df) < len(large_sample_df)
    assert len(sampler.df[sampler.df["CYP2D6_Substrate"] == 0]) == len(
        sampler.df[sampler.df["CYP2D6_Substrate"] == 1]
    )


def test_class_balancing_disable(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    sampler.enable_class_balancing("CYP2D6_Substrate")
    assert len(sampler.df) < len(large_sample_df)

    sampler.disable_class_balancing()
    assert len(sampler.df) == len(large_sample_df)
    assert (
        sampler.df["CYP2D6_Substrate"].value_counts()
        != sampler.df["CYP2D6_Substrate"].value_counts().iloc[0]
    ).any()


def test_continuous_value_formatting(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    template = "The {LogP__names__noun} of {compound_name#} is {LogP#} {LogP__units}."
    result = sampler.sample(large_sample_df.iloc[0], template)

    assert "LogP value" in result["text"] or "partition coefficient" in result["text"]
    assert "log units" in result["text"]
    assert re.search(
        r"\d+\.\d{2} log units", result["text"]
    )  # Check if the value is rounded to 2 decimal places


def test_error_handling_invalid_variable(sample_df, sample_meta, sample_config):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config)
    template = "This is an {invalid_variable#}."

    with pytest.raises(KeyError):
        sampler.sample(sample_df.iloc[0], template)


def test_multiple_targets_in_template(
    large_sample_df, large_sample_meta, sample_config
):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    template = "The molecule {compound_name#} with {SMILES__description} {SMILES#} has a {LogP__names__noun} of {LogP#} {LogP__units} and is {CYP2D6_Substrate#not &NULL}a {CYP2D6_Substrate__names__noun}."
    result = sampler.sample(large_sample_df.iloc[0], template)
    assert all(x in result["text"] for x in ["Compound_", "C", "log units", "CYP"])
    assert ("is a" in result["text"] and "not a" not in result["text"]) or (
        "is not a" in result["text"] and "is a" not in result["text"]
    )


def test_random_sampling(large_sample_df, large_sample_meta, sample_config):
    sampler = TemplateSampler(large_sample_df, large_sample_meta, sample_config)
    template = "The {compound_name#} has a {LogP__names__noun} of {LogP#}."

    # Sample multiple times without specifying a row
    results = [sampler.sample(None, template)["text"] for _ in range(10)]

    # Check if we have at least two different results (high probability)
    assert len(set(results)) > 1


def test_multiple_identifier_types(sample_df, sample_meta, sample_config_with_wrapping):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config_with_wrapping)
    template = "SMILES: {SMILES#}, Name: {compound_name#}"
    result = sampler.sample(sample_df.iloc[0], template)
    assert all(
        tag in result["text"]
        for tag in ["[BEGIN_SMILES]", "[END_SMILES]", "[BEGIN_Other]", "[END_Other]"]
    )


def test_wrapping_with_multiple_choice(
    sample_df, sample_meta, sample_config_with_wrapping
):
    sampler = TemplateSampler(sample_df, sample_meta, sample_config_with_wrapping)
    template = """
    Which compound has this SMILES: {SMILES#}?
    {%multiple_choice_enum%2%aA1}
    {compound_name%}
    Answer: {%multiple_choice_result}
    """
    result = sampler.sample(sample_df.iloc[0], template)
    assert "[BEGIN_SMILES]" in result["text"] and "[END_SMILES]" in result["text"]
    assert (
        "A or B" in result["text"]
        or "a or b" in result["text"]
        or "1 or 2" in result["text"]
    )


def test_wrapping_with_continuous_value(
    large_sample_df, large_sample_meta, sample_config_with_wrapping
):
    sampler = TemplateSampler(
        large_sample_df, large_sample_meta, sample_config_with_wrapping
    )
    template = "SMILES: {SMILES#}, LogP: {LogP#}"
    result = sampler.sample(large_sample_df.iloc[0], template)
    assert "[BEGIN_SMILES]" in result["text"] and "[END_SMILES]" in result["text"]
    assert re.search(r"LogP: \d+\.\d{2}", result["text"])  # Checks for 2 decimal places


def test_polymer_template_1(
    sample_polymer_df, sample_polymer_meta, sample_polymer_config
):
    sampler = TemplateSampler(
        sample_polymer_df, sample_polymer_meta, sample_polymer_config
    )
    template = "The polymer with the {PSMILES__description} of {PSMILES#} has an experimental glass transition temperature of {Tg_exp#} {Tg_exp__units}."
    result = sampler.sample(sample_polymer_df.iloc[0], template)
    assert "PSMILES representation" in result["text"]
    assert "*CC(*)C" in result["text"]
    assert "273.15" in result["text"]
    assert "K" in result["text"]


def test_polymer_template_2(
    sample_polymer_df, sample_polymer_meta, sample_polymer_config
):
    sampler = TemplateSampler(
        sample_polymer_df, sample_polymer_meta, sample_polymer_config
    )
    template = "The polymer with the {compound_name__names__noun} of {compound_name#} has a computed density at 300 K of {rho_300K_calc#} {rho_300K_calc__units}."
    result = sampler.sample(sample_polymer_df.iloc[1], template)
    assert "polymer name" in result["text"] or "compound name" in result["text"]
    assert "Poly(isobutylene)" in result["text"]
    assert "0.92" in result["text"]
    assert "g/cm³" in result["text"]


def test_polymer_question_answer(
    sample_polymer_df, sample_polymer_meta, sample_polymer_config
):
    sampler = TemplateSampler(
        sample_polymer_df, sample_polymer_meta, sample_polymer_config
    )
    template = """Question: What is a polymer with a computed glass transition temperature of {Tg_calc#} {Tg_calc__units} and a computed density at 300 K of {rho_300K_calc#} {rho_300K_calc__units}.

Answer: A polymer with {PSMILES__description} {PSMILES#}"""
    result = sampler.sample(sample_polymer_df.iloc[0], template)
    assert "275.0" in result["text"]
    assert "0.90" in result["text"]
    assert "PSMILES representation" in result["text"]
    assert "*CC(*)C" in result["text"]


def test_polymer_multiple_choice(
    sample_polymer_df, sample_polymer_meta, sample_polymer_config
):
    sampler = TemplateSampler(
        sample_polymer_df, sample_polymer_meta, sample_polymer_config
    )
    template = """Task: Please answer the multiple choice question.

Question: Which polymer has an experimental glass transition temperature of {Tg_exp#} {Tg_exp__units}?

Options:

{%multiple_choice_enum%3%aA1}

{compound_name%}

Answer: {%multiple_choice_result}"""
    result = sampler.sample(sample_polymer_df.iloc[0], template)
    assert "273.15" in result["text"]
    assert "K" in result["text"]
    assert any(
        symbol in result["text"]
        for symbol in ["A", "B", "C", "a", "b", "c", "1", "2", "3"]
    )

    # check that the answer is the correct polymer name, i.e. Poly(propylene)
    last_line_enum = result["text"].split("\n")[-1].replace("Answer: ", "").strip()

    # find the option with that enum
    for line in result["text"].split("\n"):
        if line.startswith(last_line_enum):
            # if any polymer name is in the line, we run the assert
            if any(
                polymer_name in line
                for polymer_name in [
                    "Poly(propylene)",
                    "Poly(ethylene)",
                    "Poly(propylene-alt-ethylene)",
                ]
            ):
                assert "Poly(propylene)" in line


def test_polymer_property_comparison(
    sample_polymer_df, sample_polymer_meta, sample_polymer_config
):
    sampler = TemplateSampler(
        sample_polymer_df, sample_polymer_meta, sample_polymer_config
    )
    template = "The polymer {compound_name#} has an experimental Tg of {Tg_exp#} K and a computed Tg of {Tg_calc#} K."
    result = sampler.sample(sample_polymer_df.iloc[0], template)
    assert "Poly(propylene)" in result["text"]
    assert "273.15" in result["text"]
    assert "275.0" in result["text"]


def test_polymer_multiple_properties(
    sample_polymer_df, sample_polymer_meta, sample_polymer_config
):
    sampler = TemplateSampler(
        sample_polymer_df, sample_polymer_meta, sample_polymer_config
    )
    template = "The polymer with PSMILES {PSMILES#} has a computed Tg of {Tg_calc#} K and a computed density at 300 K of {rho_300K_calc#} g/cm³."
    result = sampler.sample(sample_polymer_df.iloc[0], template)
    assert "*CC(*)C" in result["text"]
    assert "275.0" in result["text"]
    assert "0.90" in result["text"]


def test_sample_with_random_replacement(
    sample_multiple_identifier_df, sample_multiple_identifier_meta, sample_config
):
    sampler = TemplateSampler(
        sample_multiple_identifier_df, sample_multiple_identifier_meta, sample_config
    )
    template = "The compound with {SMILES__description} {SMILES#} has a {LogP__description} of {LogP#}"
    results = [
        sampler.sample(sample_multiple_identifier_df.iloc[0], template)["text"]
        for _ in range(20)
    ]
    smiles_count = sum("CC(C)NCC(O)c1ccc(O)c(O)c1" in r for r in results)
    selfies_count = sum(
        "[C][C][Branch1][C][C][N][C][C][Branch1][O][C][=C][C][=C][C][Branch1][O][C][=C][Branch1][O][C][=C]"
        in r
        for r in results
    )
    inchi_count = sum(
        "InChI=1S/C11H17NO3/c1-8(2)12-6-9(13)10-4-3-5-11(14)7-10/h3-5,7-9,12-14H,6H2,1-2H3"
        in r
        for r in results
    )
    assert smiles_count > 0
    assert selfies_count > 0
    assert inchi_count > 0
    assert smiles_count + selfies_count + inchi_count == 20


def test_benchmarking_template(sample_df, sample_meta, sample_config):
    config = sample_config.copy()
    config["benchmarking_templates"] = True
    sampler = TemplateSampler(sample_df, sample_meta, config)
    template = "The molecule with SMILES {SMILES#} has a CYP2D6 substrate status of:<EOI>{CYP2D6_Substrate#}"
    result = sampler.sample(sample_df.iloc[0], template)
    assert "input" in result and "output" in result
    assert "CC(C)NCC(O)c1ccc(O)c(O)c1" in result["input"]
    assert result["output"] in ["0", "1"]


def test_multiple_choice_benchmarking_template(sample_df, sample_meta, sample_config):
    config = sample_config.copy()
    config["benchmarking_templates"] = True
    config["multiple_choice_benchmarking_templates"] = True
    sampler = TemplateSampler(sample_df, sample_meta, config)
    template = """Question: Is the molecule with SMILES {SMILES#} a CYP2D6 substrate?
{%multiple_choice_enum%2%aA1}
{CYP2D6_Substrate%}
<EOI>{%multiple_choice_result}"""
    result = sampler.sample(sample_df.iloc[0], template)
    assert "input" in result and "output" in result
    assert "CC(C)NCC(O)c1ccc(O)c(O)c1" in result["input"]
    assert result["answer_choices"]
    assert result["correct_output_index"]


def test_additional_targets_handling(
    sample_multiple_identifier_df, sample_multiple_identifier_meta, sample_config
):
    sampler = TemplateSampler(
        sample_multiple_identifier_df, sample_multiple_identifier_meta, sample_config
    )
    template = (
        "The molecule with {SMILES__description} {SMILES#} has a LogP of {LogP#}."
    )
    results = [
        sampler.sample(sample_multiple_identifier_df.iloc[0], template)
        for _ in range(20)
    ]
    identifiers = ["SMILES", "SELFIES", "InChI"]
    counts = {}
    for result in results:
        for identifier in identifiers:
            if identifier in result["text"]:
                counts[identifier] = counts.get(identifier, 0) + 1
    assert all(count > 0 for count in counts.values())
    assert sum(counts.values()) == 20
