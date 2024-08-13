from litellm import completion
import yaml
import fire

CONSTANT_PROMPT_FOR_TEMPLATE_AUGMENTATION = """

Guidelines for advanced templates:

1. Multiple Choice Questions:
- Use the format {%multiple_choice_enum%N%TYPE} where N is the number of choices and TYPE is the enumeration style (e.g., aA1 for lowercase, uppercase, or numbers).
- Use {COLUMN_NAME%} to indicate where the choices should be inserted.
- Use {%multiple_choice_result} for the correct answer.
For example `Task: Please answer the multiple choice question.
Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {CYP2D6_Substrate__names__verb}?
Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
Options:
{CYP2D6_Substrate%}
Answer: {%multiple_choice_result}`

2. Benchmarking Templates:
- Include the <EOI> tag to separate the question from the answer.
- These templates are used for evaluating model performance. For example,
`Is the {SMILES__description} {SMILES#} a {CYP2D6_Substrate__names__noun}:<EOI>{CYP2D6_Substrate#no&yes}`

3. Conditional Statements:
- Use {COLUMN#not &NULL} for conditional text based on column values.

4. Random Choices:
- Use {#option1|option2|option3!} for random selection of text.

5. Use `__` to access a field in the meta data. for example {target__names__noun} will access the noun field of the target names.

Generate a similar meta.yaml structure for the given dataset, including appropriate targets, identifiers, and templates based on the column names and example data provided. If you are unsure, do not perform any augmentation. Try to add at least five new templates. Do not manually insert enumeration symbols (such a, b, c), units or identifier names. Take those from the yaml file.

Ensure that the sampled text will be valid English and valid in the context of the data. Also be sure that the questions are meaningful from a chemical point of view and that the answer is not trivially inferable from the question.

Just return raw YAML string, do not wrap it into backticks or anything else.
"""


def generate_augmented_meta_yaml(
    data_dir: str, output_dir: str, model_name: str = "gpt-4o"
):
    """
    Generate augmented meta.yaml for the given dataset.

    Args:
        data_dir (str): Path to directory containing meta.yaml and data_clean.csv
        output_dir (str): Path to directory for output files

    Returns:
        str: Augmented meta.yaml
    """

    with open(f"{data_dir}/meta.yaml", "r") as f:
        yaml_content = f.read()

    prompt = f"""You are given the following meta.yaml structure for a dataset. Based on the column names and example data provided, augment the meta.yaml structure with appropriate templates. Ensure that the logic in the templates is consistent with the data and existing templates. If you are unsure, do not perform any augmentation.

    {yaml_content}
    """

    llm_response = completion(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt + CONSTANT_PROMPT_FOR_TEMPLATE_AUGMENTATION,
            },
        ],
        temperature=0,
    )

    llm_response = llm_response.choices[0].message.content

    # Parse the LLM's response and convert it to a dictionary
    try:
        meta_yaml = yaml.safe_load(llm_response)
    except yaml.YAMLError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"LLM response: {llm_response}")
        meta_yaml = None

    return meta_yaml


def cli(data_dir: str, model: str = "gpt-4o", override: bool = False):
    """
    Generate augmented meta.yaml for the given dataset.

    Args:
        data_dir (str): Path to directory containing meta.yaml and data_clean.csv
        model (str): Model name to use for completion
        override (bool): Whether to override the existing meta.yaml file

    Returns:
        str: Augmented meta.yaml
    """

    augmented_meta_yaml = generate_augmented_meta_yaml(data_dir, model)

    if augmented_meta_yaml:
        if override:
            with open(f"{data_dir}/meta.yaml", "w") as f:
                yaml.dump(augmented_meta_yaml, f)
        else:
            print(yaml.dump(augmented_meta_yaml))

    return augmented_meta_yaml


if __name__ == "__main__":
    fire.Fire(cli)
