import pytest
import os
import pandas as pd
import yaml
import json
from chemnlp.data.sampler_cli import process_dataset

@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create meta.yaml
    meta = {
        'identifiers': [{'id': 'SMILES', 'type': 'SMILES'}],
        'targets': [{'id': 'property', 'type': 'continuous'}],
        'templates': [
            'The molecule with SMILES {SMILES#} has property {property#}.',
            'What is the property of the molecule with SMILES {SMILES#}?<EOI>{property#}'
        ]
    }
    with open(data_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f)

    # Create data_clean.csv
    df = pd.DataFrame({
        'SMILES': ['CC', 'CCC', 'CCCC'],
        'property': [1.0, 2.0, 3.0],
        'split': ['train', 'test', 'valid']
    })
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
        multiple_choice=False
    )

    # Check that output files were created
    chunk_dir = temp_output_dir / "chunk_0"
    template_dir = chunk_dir / "template_0"
    assert template_dir.exists()

    # Check the content of the output files
    for split in ['train', 'test', 'valid']:
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
        multiple_choice=False
    )

    # Check that output files were created
    chunk_dir = temp_output_dir / "chunk_0"
    template_dir = chunk_dir / "template_0"
    assert template_dir.exists()

    # Check the content of the output files
    for split in ['train', 'test', 'valid']:
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
        multiple_choice=False
    )

    # Check that output files were created
    chunk_dir = temp_output_dir / "chunk_0"
    template_dir = chunk_dir / "template_0"
    assert template_dir.exists()

    # Check the content of the output files
    for split in ['train', 'test', 'valid']:
        with open(template_dir / f"{split}.jsonl", "r") as f:
            lines = f.readlines()
            assert len(lines) == 1  # One sample per split
            sample = json.loads(lines[0])
            assert "text" in sample
            assert "SMILES" in sample["text"]
            assert "property" in sample["text"]
