# Sampler Module

## Overview

The `sampler` module provides functionality for generating text samples based on templates and data. It is primarily used for creating datasets for natural language processing tasks in chemistry and related fields. The main class in this module is `TemplateSampler`, which allows for flexible text generation with support for multiple choice questions, class balancing, and identifier wrapping.

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

#### Configuration Options

- `wrap_identifiers`: Boolean flag to enable wrapping of identifiers with tags (default: False).

#### Main Methods

##### `sample`

```python
def sample(self, sample: Optional[pd.Series], template: str) -> str
```

Generates a text sample based on a template and a data sample.

- `sample`: A row from the dataset. If None, a random sample is chosen.
- `template`: The template string to be filled.
- Returns: The completed text sample with all variables replaced by their values.

##### `enable_class_balancing`

```python
def enable_class_balancing(self, column: str)
```

Enables class-balanced sampling for a specified column.

- `column`: The column to use for balancing.

##### `disable_class_balancing`

```python
def disable_class_balancing(self)
```

Disables class-balanced sampling and reverts to the original dataset.

#### Identifier Wrapping

When `wrap_identifiers` is set to `True` in the configuration, the sampler will wrap identifier values with tags. For example:

- `[BEGIN_SMILES]CC(C)NCC(O)c1ccc(O)c(O)c1[END_SMILES]`
- `[BEGIN_InChI]InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)[END_InChI]`

This feature can be useful for downstream tasks that need to identify specific types of chemical identifiers in the generated text.

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


Basic usage with identifier wrapping:

```python
import pandas as pd
from chemnlp.data.sampler import TemplateSampler

# Prepare your data, metadata, and config
df = pd.DataFrame(...)
meta = {...}
config = {
    'wrap_identifiers': True,
    # ... other config options
}

# Initialize the sampler
sampler = TemplateSampler(df, meta, config)

# Define a template
template = "The molecule with SMILES {SMILES#} has a {property#} of {value#}."

# Generate a sample
result = sampler.sample(df.iloc[0], template)
print(result)
# Output: The molecule with SMILES [BEGIN_SMILES]CC(C)NCC(O)c1ccc(O)c(O)c1[END_SMILES] has a LogP of 1.23.
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

- The `TemplateSampler` class supports wrapping of identifiers with tags when the `wrap_identifiers` option is enabled in the configuration.
- Wrapped identifiers use the format `[BEGIN_IDENTIFIER_TYPE]value[END_IDENTIFIER_TYPE]`.
- Identifier types are based on the `IdentifierEnum` class, which includes common chemical identifiers like SMILES, InChI, and others.

For more detailed information on the implementation and advanced usage, please refer to the source code and unit tests.
