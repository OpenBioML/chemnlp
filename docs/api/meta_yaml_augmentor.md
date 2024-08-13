# Meta YAML Augmenter

## Overview

The Meta YAML Augmenter is a tool designed to enhance existing `meta.yaml` files for chemical datasets. It uses Large Language Models (LLMs) to generate additional templates and improve the metadata structure, particularly focusing on advanced sampling methods and template formats.

## generate_augmented_meta_yaml

::: chemnlp.data.meta_yaml_augmenter.generate_augmented_meta_yaml
handler: python
options:
show_root_heading: true
show_source: false

## CLI Interface

The module provides a command-line interface for easy augmentation of `meta.yaml` files.

### Usage

```bash
python -m chemnlp.data.meta_yaml_augmenter <data_dir> [--model MODEL] [--override]
```

### Arguments

- `data_dir` (str): Path to the directory containing the `meta.yaml` file to be augmented.
- `--model` (str, optional): The name of the LLM model to use for augmentation. Default is 'gpt-4o'.
- `--override` (flag): If set, the existing `meta.yaml` file will be overwritten with the augmented version.

### Example

```bash
python -m chemnlp.data.meta_yaml_augmenter /path/to/dataset --model gpt-4o --override
```

## Augmentation Process

The augmentation process involves:

1. Reading the existing `meta.yaml` file from the specified directory.
2. Sending the content to an LLM along with guidelines for creating advanced templates.
3. Parsing the LLM's response to generate an augmented `meta.yaml` structure.
4. Either printing the augmented structure or overwriting the existing file, based on the `override` flag.

## Notes

1. **LLM Integration**: This tool requires integration with an LLM service. Ensure you have the necessary credentials and access set up. By default it uses, `gpt-4o`. For this, you need to expose the `OPENAI_API_KEY` environment variable.

2. **Output Quality**: The quality of the augmented `meta.yaml` depends on the capabilities of the LLM being used. Manual review and adjustment may be necessary.
   It also depends on the quality of the existing `meta.yaml` file. If the existing file doesn't follow the standards (and, for example, hard codes target names) the augmentation may not be successful.

## Example Usage in Python

```python
from chemnlp.data.meta_yaml_augmenter import generate_augmented_meta_yaml

data_dir = "/path/to/dataset"
model_name = "gpt-4o"

augmented_yaml = generate_augmented_meta_yaml(data_dir, model_name)

if augmented_yaml:
    print(yaml.dump(augmented_yaml))
```
