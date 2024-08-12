# Sampler Module

## Overview

The `sampler` module provides functionality for generating text samples based on templates and data. It is primarily used for creating datasets for natural language processing tasks in chemistry and related fields. The main class in this module is `TemplateSampler`, which allows for flexible text generation with support for multiple choice questions and class balancing.

## TemplateSampler

### Class: TemplateSampler

The `TemplateSampler` class is responsible for sampling and generating text based on templates and data.

#### Initialization

```python
sampler = TemplateSampler(df: pd.DataFrame, meta: Dict, config: Dict, column_datafield_sampler: Optional[Callable] = None)
```

- `df`: A pandas DataFrame containing the dataset.
- `meta`: A dictionary containing metadata about the dataset, including identifiers and targets.
- `config`: A dictionary containing configuration parameters for the sampler.
- `column_datafield_sampler`: An optional callable for custom sampling from multiple options.

#### Main Methods

##### sample

```python
def sample(self, sample: Optional[pd.Series], template: str) -> str
```

Generates a text sample based on a template and a data sample.

- `sample`: A row from the dataset. If None, a random sample is chosen.
- `template`: The template string to be filled.
- Returns: The completed text sample with all variables replaced by their values.

##### enable_class_balancing

```python
def enable_class_balancing(self, column: str)
```

Enables class-balanced sampling for a specified column.

- `column`: The column to use for balancing.

##### disable_class_balancing

```python
def disable_class_balancing(self)
```

Disables class-balanced sampling and reverts to the original dataset.

#### Usage Examples

Basic usage:

```python
import pandas as pd
from chemnlp.data.sampler import TemplateSampler

# Prepare your data, metadata, and config
df = pd.DataFrame(...)
meta = {...}
config = {...}

# Initialize the sampler
sampler = TemplateSampler(df, meta, config)

# Define a template
template = "The molecule with SMILES {SMILES#} has a {property#} of {value#}."

# Generate a sample
result = sampler.sample(df.iloc[0], template)
print(result)
```

Using class balancing:

```python
# Enable class balancing
sampler.enable_class_balancing("target_column")

# Generate balanced samples
balanced_results = [sampler.sample(None, template) for _ in range(100)]

# Disable class balancing when done
sampler.disable_class_balancing()
```

Multiple choice question:

```python
multiple_choice_template = """
Question: What is the {property__names__noun} of the molecule with SMILES {SMILES#}?
Options: {%multiple_choice_enum%4%aA1}
{value%}
Answer: {%multiple_choice_result}
"""

mc_result = sampler.sample(df.iloc[0], multiple_choice_template)
print(mc_result)
```

## Notes

- The `TemplateSampler` class supports various types of templates, including those with multiple choice questions.
- Class balancing can be useful for creating balanced datasets for machine learning tasks.
- The sampler can handle both categorical and continuous data types, with proper formatting for continuous values.
- Custom sampling functions can be provided for more control over how values are selected from multiple options.

For more detailed information on the implementation and advanced usage, please refer to the source code and unit tests.
