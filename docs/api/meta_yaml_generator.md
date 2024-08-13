# Meta YAML Generator

## Overview

The Meta YAML Generator is a tool designed to automatically create a `meta.yaml` file for chemical datasets using Large Language Models (LLMs). It analyzes the structure of a given DataFrame and generates a comprehensive metadata file, including advanced sampling methods and template formats.

The model used by default is `gpt4o`. For using it, you need to expose the `OPENAI_API_KEY` environment variable.

## `generate_meta_yaml`

::: chemnlp.data.meta_yaml_generator.generate_meta_yaml
handler: python
options:
show_root_heading: true
show_source: false

## Usage Example

```python
import pandas as pd
from chemnlp.data.meta_yaml_generator import generate_meta_yaml

# Load your dataset
df = pd.read_csv("your_dataset.csv")

# Generate meta.yaml
meta_yaml = generate_meta_yaml(
    df,
    dataset_name="Polymer Properties Dataset",
    description="A dataset of polymer properties including glass transition temperatures and densities",
    output_path="path/to/save/meta.yaml"
)

# The meta_yaml variable now contains the dictionary representation of the meta.yaml
# If an output_path was provided, the meta.yaml file has been saved to that location
```

You can also use it as a command-line tool:

```bash
python -m chemnlp.data.meta_yaml_generator path/to/your_dataset.csv --dataset_name "Polymer Properties Dataset" --description "A dataset of polymer properties including glass transition temperatures and densities" --output_path "path/to/save/meta.yaml"
```
