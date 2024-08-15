import pytest
import pandas as pd
import yaml
import json
from chemnlp.data.sampler_cli import process_dataset
from chemnlp.data.constants import STANDARD_TABULAR_TEXT_TEMPLATES


@pytest.fixture
def temp_tabular_data_dir(tmp_path):
    data_dir = tmp_path / "tabular" / "test_dataset"
    data_dir.mkdir(parents=True)

    # Create meta.yaml
    meta = {
        "identifiers": [{"id": "SMILES", "type": "SMILES", "description": "SMILES"}],
        "targets": [
            {
                "id": "logP",
                "type": "continuous",
                "names": [{"noun": "logP"}],
                "units": "",
            }
        ],
        "templates": [
            "Custom template: The molecule with SMILES {SMILES#} has logP {logP#}.",
        ],
    }
    with open(data_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f)

    # Create data_clean.csv
    df = pd.DataFrame(
        {
            "SMILES": ["CC", "CCC", "CCCC"],
            "logP": [1.0, 2.0, 3.0],
            "split": ["train", "test", "valid"],
        }
    )
    df.to_csv(data_dir / "data_clean.csv", index=False)

    return data_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create meta.yaml
    meta = {
        "identifiers": [{"id": "SMILES", "type": "SMILES"}],
        "targets": [{"id": "property", "type": "continuous"}],
        "templates": [
            "The molecule with SMILES {SMILES#} has property {property#}.",
            "What is the property of the molecule with SMILES {SMILES#}?<EOI>{property#}",
        ],
    }
    with open(data_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f)

    # Create data_clean.csv
    df = pd.DataFrame(
        {
            "SMILES": ["CC", "CCC", "CCCC"],
            "property": [1.0, 2.0, 3.0],
            "split": ["train", "test", "valid"],
        }
    )
    df.to_csv(data_dir / "data_clean.csv", index=False)

    return data_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


def test_process_dataset(temp_data_dir, temp_output_dir):
    process_dataset(
        data_dir=str(temp_data_dir),
        output_dir=str(temp_output_dir),
        chunksize=1000,
        class_balanced=False,
        benchmarking=False,
    )

    # Check that output files were created
    chunk_dir = temp_output_dir / "data" / "chunk_0"
    template_dir = chunk_dir / "template_0"
    assert template_dir.exists()

    # Check the content of the output files
    for split in ["train", "test", "valid"]:
        with open(template_dir / f"{split}.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 1  # One sample per split
            sample = json.loads(lines[0])
            assert "text" in sample
            assert "SMILES" in sample["text"]
            assert "property" in sample["text"]


def test_process_dataset_benchmarking(temp_data_dir, temp_output_dir):
    process_dataset(
        data_dir=str(temp_data_dir),
        output_dir=str(temp_output_dir),
        chunksize=1000,
        class_balanced=False,
        benchmarking=True,
    )

    # Check that output files were created
    chunk_dir = temp_output_dir / "data" / "chunk_0"
    template_dir = chunk_dir / "template_0"
    assert template_dir.exists()

    # Check the content of the output files
    for split in ["train", "test", "valid"]:
        with open(template_dir / f"{split}.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 1  # One sample per split
            sample = json.loads(lines[0])
            assert "input" in sample
            assert "output" in sample
            assert "SMILES" in sample["input"]
            # assert that we can convert the output to a float
            try:
                float(sample["output"])
            except ValueError:
                assert False


def test_process_dataset_class_balanced(temp_data_dir, temp_output_dir):
    process_dataset(
        data_dir=str(temp_data_dir),
        output_dir=str(temp_output_dir),
        chunksize=1000,
        class_balanced=True,
        benchmarking=False,
    )

    # Check that output files were created
    chunk_dir = temp_output_dir / "data" / "chunk_0"
    template_dir = chunk_dir / "template_0"
    assert template_dir.exists()

    # Check the content of the output files
    for split in ["train", "test", "valid"]:
        with open(template_dir / f"{split}.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 1  # One sample per split
            sample = json.loads(lines[0])
            assert "text" in sample
            assert "SMILES" in sample["text"]
            assert "property" in sample["text"]


def test_process_dataset_with_standard_templates(
    temp_tabular_data_dir, temp_output_dir
):
    process_dataset(
        data_dir=str(temp_tabular_data_dir),
        output_dir=str(temp_output_dir),
        chunksize=1000,
        class_balanced=False,
        benchmarking=False,
        use_standard_templates=True,
    )

    # Check that output files were created
    chunk_dir = temp_output_dir / "test_dataset" / "chunk_0"

    # Count the number of template directories
    template_dirs = list(chunk_dir.glob("template_*"))

    # Expected number of templates: 1 custom + len(STANDARD_TABULAR_TEXT_TEMPLATES)
    expected_template_count = 1 + len(
        [t for t in STANDARD_TABULAR_TEXT_TEMPLATES if "<EOI>" not in t]
    )
    assert (
        len(template_dirs) == expected_template_count
    ), f"Expected {expected_template_count} templates, but found {len(template_dirs)}"

    # Check the content of the output files for each template
    for template_dir in template_dirs:
        for split in ["train", "test", "valid"]:
            with open(template_dir / f"{split}.jsonl", "r") as f:
                lines = f.readlines()
                assert len(lines) == 1  # One sample per split
                sample = json.loads(lines[0])
                assert "text" in sample
                assert "SMILES" in sample["text"]
                assert "logP" in sample["text"]
