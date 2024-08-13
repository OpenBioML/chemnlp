# Contributing to ChemNLP

Thank you for your interest in contributing to ChemNLP! There are many ways to contribute, including implementing datasets, improving code, and enhancing documentation.

## Getting Started

1. Create a [GitHub account](https://github.com/) if you don't have one.
2. [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the [ChemNLP repository](https://github.com/OpenBioML/chemnlp).
3. [Clone your fork](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).
4. [Create a new branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) for your contribution.
5. Set up your development environment as described in the `Installation and set-up` section of [README.md](README.md).

## Implementing a Dataset

One of the most valuable contributions is implementing a dataset. Here's how to do it:

1. Choose a dataset from our [awesome list](https://github.com/kjappelbaum/awesome-chemistry-datasets) or add a new one there.
2. Create an issue in this repository stating your intention to add the dataset.
3. Make a Pull Request (PR) that adds a new folder in `data` with the following files:

   - `meta.yaml`: Describes the dataset (see structure below).
   - `transform.py`: Python code to transform the original dataset into a usable form.

### meta.yaml Structure

```yaml
name: dataset_name
description: Short description of the dataset
targets:
  - id: target_name
    description: Description of the target
    units: Units of the target (if applicable)
    type: continuous or boolean
    names:
      - noun: target noun
      - adjective: target adjective
benchmarks:
  - name: benchmark_name
    link: benchmark_link
    split_column: split
identifiers:
  - id: identifier_name
    type: SMILES, InChI, etc.
    description: Description of the identifier
license: Dataset license
num_points: Number of datapoints
links:
  - name: link_name
    url: link_url
    description: Link description
bibtex: Citation in BibTeX format
```

### transform.py Guidelines

- Download data from an official source or upload it to a repository and retrieve it from there.
- For tabular datasets: remove/merge duplicates, rename columns, and drop unused columns.
- Output should be as lean as possible, typically in a `data_clean.csv` file.
- Add any necessary dependencies to `dev-requirements.txt` or `requirements.txt`.

## Text Templates

Text templates are used for sampling and can utilize data from `meta.yaml`, recode categorical data, and chain multiple data fields. Examples include:

1. Basic template:

   ```
   The molecule with {SMILES__description} {SMILES#} has {property#} {property__units}.
   ```

2. Multiple choice template:

   ```
   Task: Answer the multiple choice question.
   Question: Is the molecule with {SMILES__description} {SMILES#} {property__names__adjective}?
   Options: {%multiple_choice_enum%2%aA1}
   {property%}
   Answer: {%multiple_choice_result}
   ```

3. Benchmarking template:
   ```
   Is the molecule with {SMILES__description} {SMILES#} {property__names__adjective}?<EOI>{property#yes&no}
   ```

## Testing Your Contribution

- Ensure your code passes all existing tests.
- Add new tests for any new functionality you introduce.
- Run `pytest` to check all tests pass.

## Submitting Your Contribution

1. Commit your changes using [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/).
2. Push your changes to your fork.
3. Create a Pull Request to the main ChemNLP repository.
4. Respond to any feedback on your PR.

Thank you for contributing to ChemNLP! Your efforts help advance chemical natural language processing research and applications.
