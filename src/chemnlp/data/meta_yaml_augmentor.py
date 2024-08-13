from litellm import completion
import yaml
import fire

CONSTANT_PROMPT_FOR_TEMPLATE_AUGMENTATION = """

Guidelines for templates:

1. Multiple Choice Questions:
- Use the format {%multiple_choice_enum%N%TYPE} where N is the number of choices and TYPE is the enumeration style (e.g., aA1 for lowercase, uppercase, or numbers).
- Use {COLUMN_NAME%} to indicate where the choices should be inserted.
- Use {%multiple_choice_result} for the correct answer. For example,

    - `Task: Please answer the multiple choice question.
    Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {CYP2D6_Substrate__names__verb}?
    Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
    Options:
    {CYP2D6_Substrate%}
    Answer: {%multiple_choice_result}`

2. Benchmarking Templates:
- This is crucial to include some questions in this style
- The <EOI> tag is supposed to separate the question from the answer. That is, the question MUST be before the <EOI> tag and the answer should be after the <EOI> tag. This is something one can do for most questions.
- These templates are used for evaluating model performance. For example,

    - `Is the {SMILES__description} {SMILES#} a {CYP2D6_Substrate__names__noun}:<EOI>{CYP2D6_Substrate#no&yes}`
    - `Task: Please {#create|generate!} a {#molecule |!}{SMILES__description} that has a {#bioaffinity|affinity!} to {#the protein |!}{protein_name#} with a {standard_type#} {#value |!}of {standard_value#} {standard_units#}.
    Result:<EOI>{SMILES#}`
    - `Task: Please {#derive|estimate|approximate!} {#the bioaffinity|the affinity!} of a {#molecule to a protein|protein to a molecule!}.
    Protein{# name|!}: {protein_name#}
    {#Molecule |!}{SMILES__description}: {SMILES#}
    Constraint{#s|!}: The {#resulting|derived|calculated!} {standard_type#} {#value |!}should be in {standard_units#}. Even if you are {#uncertain|not sure!}, you must {#derive|estimate|come up with!} a {standard_type#} {#value |!}without using any {#other|additional!} words.
    Result:<EOI>{standard_value#} {standard_units#}`
    - `Question: What is the {#the bioaffinity|the affinity!} of a {#molecule to a protein|protein to a molecule!}?
    Protein{# name|!}: {protein_name#}
    {#Molecule |!}{SMILES__description}: {SMILES#}
    Constraint: The {#shown|listed!} {standard_type#} values {#below |!}are in {standard_units#}. Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%3-5%aA1} without using any other words.
    Options:
    {standard_value%}
    Answer:<EOI>{%multiple_choice_result}`

3. Conditional Statements:
- Use {COLUMN#not &NULL} for conditional text based on column values. Note that this only makes sense for columns that are boolean.

4. Random Choices:
- Use {#option1|option2|option3!} for random selection of text. Use this to add variety to the prompts. If there are synonyms or different ways to ask the same question, use this to add variety. For example

    - `Task: Please {#create|generate!} a {#molecule |!}{SMILES__description} that has a {#bioaffinity|affinity!} to {#the protein |!}{protein_name#} with a {standard_type#} {#value |!}of {standard_value#} {standard_units#}.
    Result:<EOI>{SMILES#}`

5. Use `__` to access a field in the meta data. for example {target__names__noun} will access the noun field of the target names. NEVER hardcode the target names. Always use the templating syntax. This is also crucial for identifiers as we have some post-processing routines that rely on this.

Generate a similar meta.yaml structure for the given dataset, including appropriate targets, identifiers, and templates based on the column names and example data provided. If you are unsure, do not perform any augmentation. Try to add at least five new templates, including some in benchmarking style. Do not manually insert enumeration symbols (such a, b, c), units or identifier names. Take those from the yaml file.

Ensure that the sampled text will be valid English and valid in the context of the data. Also be sure that the questions are meaningful from a chemical point of view and that the answer is not trivially inferable from the question. Ideally, aim for a diversity in style to cover many possible ways people might make statements or ask questions about the data. You can also include statements or user/assistant interactions instead of questions.

Ensure that you include some questions with the <EOI> tag, we MUST have those for benchmarking purposes.

You must use the templating syntax to insert variable, identifier and tagret names, e.g., {protein_name__names__noun} or {SMILES__description}.Do not hardcode any values.

Include multiple choice templates. Include the correct answer in the template using the `%multiple_choice_result%` syntax and `%multiple_choice_enum%3%aA` for enumeration and `{column}` for the choices/options. For example,

    - `Question: Is the molecule with the {SMILES__description} {#representation of |!}{SMILES#} {activity_choline_transporter__names__gerund}?
    Constraint: Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%2%aA1} without using any {#other|additional!} words.
    Options:
    {activity_choline_transporter%}
    Answer:<EOI>{%multiple_choice_result}

`multiple_choice_enum` generates a string like `a,b or c`. Use it accordingly.

Ensure that you add some diversity into templates, at least via synonyms that we sample randomly. For example, use {#create|generate!} to add variety to the prompts, or `{#molecule|compound!}, or {#Assistant|Bot!}, or {#pick|choose|select!}.

Do not start lines with curly braces. Do not use curly braces for anything other than templating. Use `|-` for multi-line strings in YAML. For example,

    - |-
      This is a multi-line string.
      It can contain multiple lines.

Stick to the syntax of the templating language that is described above. Do not use any other syntax and do not remove the special characters.

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

    prompt = f"""You are given the following meta.yaml structure for a dataset. Based on the column names and example data provided, augment the meta.yaml structure with appropriate templates. Ensure that the logic in the templates is consistent with the data and existing templates. If you are unsure, do not perform any augmentation. Add at least five new templates.  Ensure to add templates in benchmarking style (e.g.., using `Answer:<EOI>` to seperate questions from the answer, ideally we want to have also questions where the answer after <EOI> is only one number or string, such as Answer:<EOI>number. You can do this sometimes by specifying in the question in which unit you want to have the answer). Ensure to use the name/description variables to insert the identifier and target names, for example using `{{target__names__noun}}`. Never hardcode those values. Do not start lines with `{{`.

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
