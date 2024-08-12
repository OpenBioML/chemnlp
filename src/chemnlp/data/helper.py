import pandas as pd
import yaml
from typing import Dict, Any
from litellm import completion

CONSTANT_PROMPT = """

Use the following example as a guide for the structure and content, paying special attention to the advanced template formats:

```yaml
name: bicerano_dataset
description: |-
This paper outlines a MD simulation workflow based on GPU MD simulation and the
refined optimized potentials for liquid simulation (OPLS) OPLS3e force field to
calculate glass transition temperatures (Tgs) of 315 polymers for which Bicerano
reported experimental values.
targets:
- id: Tg_exp
    description: experimental glass transition temperature
    units: K
    type: float
    names:
    - noun: experimental glass transition temperature
    uris:
- id: Tg_calc
    description: calculated glass transition T
    units: K
    type: float
    names:
    - noun: computed glass transition temperature
- id: rho_300K_calc
    description: computed density at 300K
    units: g/cm^3
    type: float
    names:
    - noun: computed polymer density at 300K
identifiers:
- id: PSMILES
    type: PSMILES
    description: PSMILES
- id: compound_name
    type: Other
    names:
    - noun: compound name
    description: polymer name
license: CC BY 4.0
links:
- url: https://pubs.acs.org/doi/10.1021/acsapm.0c00524#
    description: corresponding publication
- url:
    - https://raw.githubusercontent.com/AdrianM0/chemnlp/main/data/tabular/bicerano_dataset/HT_MD_polymer_properties.csv
    description: data source
num_points: 315
bibtex:
- |-
    @article{afzal2021,
    author = {Afzal, Mohammad Atif Faiz and Browning, Andrea R. and Goldberg, Alexander and Halls, Mathew D. and Gavartin, Jacob L. and Morisato,
    Tsuguo and Hughes, Thomas F. and Giesen, David J. and Goose, Joseph E.},
    title = {High-Throughput Molecular Dynamics Simulations and Validation of Thermophysical Properties of Polymers for Various Applications},
    journal = {ACS Applied Polymer Materials},
    volume = {3},
    number = {2},
    pages = {620-630},
    year = {2021},
    doi = {10.1021/acsapm.0c00524}}
templates:
- The polymer with the {PSMILES__description} of {PSMILES#} has an experimental glass transition temperature of {Tg_exp#} K.
- The polymer with the {PSMILES__description} of {PSMILES#} has a computed glass transition temperature of {Tg_calc#} K.
- The polymer with the {PSMILES__description} of {PSMILES#} has a computed density at 300 K of {rho_300K_calc#} g/cc.
- The polymer with the {compound_name__names__noun} of {compound_name#} has an experimental glass transition temperature of {Tg_exp#} K.
- The polymer with the {compound_name__names__noun} of {compound_name#} has a computed glass transition temperature of {Tg_calc#} K.
- The polymer with the {compound_name__names__noun} of {compound_name#} has a computed density at 300 K of {rho_300K_calc#} g/cc.
- |-
    Question: What is a polymer with a computed glass transition temperature of {Tg_calc#} K and a computed density at 300 K of {rho_300K_calc#} g/cc.
    Answer: A polymer with {PSMILES__description} {PSMILES#}
# Multiple choice template
- |-
    Question: Which of the following polymers has an experimental glass transition temperature closest to {Tg_exp#} K?
    Options: {%multiple_choice_enum%4%aA1}
    {compound_name%}
    Answer: {%multiple_choice_result}
# Benchmarking template
- The experimental glass transition temperature of the polymer with {PSMILES__description} {PSMILES#} is:<EOI>{Tg_exp#} K
```

Guidelines for advanced templates:

1. Multiple Choice Questions:
- Use the format {%multiple_choice_enum%N%TYPE} where N is the number of choices and TYPE is the enumeration style (e.g., aA1 for lowercase, uppercase, or numbers).
- Use {COLUMN_NAME%} to indicate where the choices should be inserted.
- Use {%multiple_choice_result} for the correct answer.

2. Benchmarking Templates:
- Include the <EOI> tag to separate the question from the answer.
- These templates are used for evaluating model performance.

3. Conditional Statements:
- Use {COLUMN#not &NULL} for conditional text based on column values.

4. Random Choices:
- Use {#option1|option2|option3!} for random selection of text.

Generate a similar meta.yaml structure for the given dataset, including appropriate targets, identifiers, and templates based on the column names and example data provided. Include at least one multiple choice template and one benchmarking template."""


def generate_meta_yaml(
    df: pd.DataFrame, dataset_name: str, description: str, model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    Generate a meta.yaml structure using an LLM based on a DataFrame, including advanced sampling methods.

    Args:
        df (pd.DataFrame): The input DataFrame.
        dataset_name (str): The name of the dataset.
        description (str): A brief description of the dataset.
        model (str, optional): The LLM model to use. Defaults to 'gpt-4o'.

    Returns:
        Dict[str, Any]: A dictionary representation of the meta.yaml structure.
    """
    # Extract column names and example data
    columns = df.columns.tolist()
    example_data = df.iloc[0].to_dict()

    # Prepare the prompt for the LLM
    prompt = (
        f"""Create a meta.yaml structure for a dataset with the following information:

Dataset name: {dataset_name}
Description: {description}
Number of rows: {len(df)}

Columns:
{', '.join(columns)}

Example data:
{yaml.dump(example_data, default_flow_style=False)}"""
        + CONSTANT_PROMPT
    )

    # Call the LLM with the prompt
    llm_response = completion(
        model=model, messages=[{"role": "user", "content": prompt}]
    )

    # Parse the LLM's response and convert it to a dictionary
    try:
        meta_yaml = yaml.safe_load(llm_response)
    except yaml.YAMLError as e:
        print(f"Error parsing LLM response: {e}")
        meta_yaml = None

    return meta_yaml


# Example usage
if __name__ == "__main__":
    # Load your DataFrame
    df = pd.read_csv("your_dataset.csv")

    # Generate meta.yaml
    meta_yaml = generate_meta_yaml(
        df,
        dataset_name="Your Dataset Name",
        description="A brief description of your dataset",
    )

    # Print or save the generated meta.yaml
    if meta_yaml:
        print(yaml.dump(meta_yaml, default_flow_style=False))

        # Optionally, save to a file
        with open("meta.yaml", "w") as f:
            yaml.dump(meta_yaml, f, default_flow_style=False)
    else:
        print("Failed to generate meta.yaml")
