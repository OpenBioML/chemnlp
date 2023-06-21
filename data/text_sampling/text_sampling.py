import glob
import math
import os.path
import random
import re
from functools import partial
from typing import Callable, List

import pandas as pd
import yaml


def unwrap_list_length_1(list_input: list):
    """Unwraps lists of length 1 and returns the first = single element."""
    if isinstance(list_input, list):
        assert len(list_input) == 1
        return list_input[0]
    else:
        raise NotImplementedError()


class RandomVariable:
    """Simple random variable class that takes in a name, data, and a sampler.
    The sampler needs to return a single element."""

    def __init__(self, name: str, data: list, sampler: Callable = None):
        self.name = name
        self.data = data
        self.sampler = partial(random.sample, k=1) if sampler is None else sampler

    def __repr__(self):
        return f"RandomVariable: {self.name}, {self.data}, {self.sampler}"

    def __call__(self) -> str:
        """Carries out sampling and returns a single element."""
        return unwrap_list_length_1(self.sampler(self.data))


def load_yaml(path: str) -> dict:
    """Load yaml file from path."""
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def get_input_variables_from_template(template: str) -> List[str]:
    """Gets all strings that are between curly brackets that are used downstream as input variables."""
    return [x for x in re.findall(r"\{([^}]+)\}", template)]


class PromptTemplate:
    """Simple prompt template class that takes a string text template as input.
    The input_variables can be defined but if not they are derived from the text template (= default use case).
    """

    def __init__(self, template: str, input_variables: List[str] = None):
        self.template = template
        if input_variables is None:
            self.input_variables = self.get_input_variables()

    def get_input_variables(self) -> List[str]:
        """Gets all variable strings from the text template that are between curly brackets
        that are used as input_variables."""
        return get_input_variables_from_template(self.template)

    def __repr__(self):
        return f"PromptTemplate: {self.template}"

    def insert(self, data: dict) -> str:
        """Inserts the data and checks before if we got all the data for the present input_variables.
        More input_variables in data doesn't matter."""
        assert all(x in data.keys() for x in self.input_variables)
        template = self.template
        for k in data:
            template = template.replace("{" + k + "}", data[k])
        return template


def get_random_text_identifiers_and_targets(meta: dict) -> dict:
    """Gets random variables as RandomVariable objects from the identifiers and targets
    in the meta dict derived from the meta.yaml."""
    rnd_texts = {}
    for e in meta["identifiers"] + meta["targets"]:
        rnd_texts[e["id"]] = {}
        if "names" in e:
            rnd_texts[e["id"]]["names"] = {}
            name_types = set([list(x.keys())[0] for x in e["names"]])
            for name in name_types:
                rnd_text = RandomVariable(
                    f"{e['id']}__names__{name}",
                    [x[name] for x in e["names"] if name in x],
                )
                rnd_texts[e["id"]]["names"][name] = rnd_text
        else:
            rnd_texts[e["id"]]["description"] = partial(
                lambda x: x, e["description"]
            )  # to wrap value in function = deterministic, no sampling

    return rnd_texts


def get_target_from_string(meta: dict, string: str) -> str:
    """Gets a target string from the meta dict based on the variable string.
    (A variable string is what is between the curly brackets in the text template.)"""
    keys = string.split("__")

    def get_with_nested_keys(d: dict, keys: list) -> str:
        t = d.copy()
        for k in keys:
            t = t[k]
        return t

    if len(keys) == 1 and keys in meta:
        return meta[
            keys
        ]  # assumes single element in meta and doesn't not support nested dicts in that version
    elif keys[0] in [x["id"] for x in meta["identifiers"] + meta["targets"]]:
        rnd_texts = get_random_text_identifiers_and_targets(meta)
        return get_with_nested_keys(rnd_texts, keys)
    else:
        raise NotImplementedError()


class TemplateSampler:
    """The template sampler uses the data_clean.csv and meta.yaml from a data directory and
    manages the the insertion of the sampled data into the text templates."""

    def __init__(
        self,
        path_data_dir: str,
        template_sampler: Callable = None,
        column_datafield_sampler: Callable = None,
        benchmarking_templates: List[str] = False,
    ):
        # paths
        self.path_data_dir = path_data_dir
        self.path_data_meta = self.path_data_dir + "/meta.yaml"
        self.path_data_csv = self.path_data_dir + "/data_clean.csv"

        # meta from yaml
        self.meta = load_yaml(self.path_data_meta)

        # dataframe from csv
        df = pd.read_csv(self.path_data_csv)

        def check_targets_and_identifiers(meta: dict, df: pd.DataFrame):
            all_identifiers = [x["id"] for x in meta["identifiers"]] + [
                x["id"] for x in meta["targets"]
            ]
            all_identifiers
            for i in all_identifiers:
                cols = df.columns.tolist()
                assert i in cols, f"target or identifier {i} not in columns {cols}!"

        check_targets_and_identifiers(self.meta, df)
        # assert not df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        if "split" not in df.columns:
            df["split"] = "full"
        self.df = df

        # text templates
        self.benchmarking_templates = benchmarking_templates
        if self.benchmarking_templates:
            self.templates = [
                t for t in self.meta["templates"] if t.find("<EOI>") != -1
            ]
        else:
            self.templates = [
                t for t in self.meta["templates"] if t.find("<EOI>") == -1
            ]
        assert self.templates is not None
        self.prompt_templates = [PromptTemplate(t) for t in self.templates]

        # create random variables for prompts and texts
        self.rnd_prompt_templates = RandomVariable(
            "rnd_prompt_templates", self.prompt_templates, template_sampler
        )
        self.rnd_texts = get_random_text_identifiers_and_targets(self.meta)

        # column_datafield_sampler
        self.column_datafield_sampler = (
            partial(random.sample, k=1)
            if column_datafield_sampler is None
            else column_datafield_sampler
        )

    def __repr__(self):
        return f"TemplateSampler: {self.path_data_dir}"

    def _get_target_from_row(self, sample: pd.Series, var: str) -> str:
        """Get target string from sample row and variable string."""
        # sampling based on columns and their definiton in the text template
        if ("#" in var) and ("&" in var):  # recoding information in var
            column, choices = var.split("#")
            choices = choices.split("&")
            choice = choices[sample[column]]
            if choice == "NULL":
                out = ""
            else:
                out = choices[sample[column]]
        elif ("#" in var) and ("|" in var):  # use data from multiple columns
            columns = var.split("|")
            columns = [var.replace("#", "") for var in columns]
            choices = sample[columns].tolist()
            choices = [c for c in choices if (isinstance(c, str) or not math.isnan(c))]
            out = unwrap_list_length_1(self.column_datafield_sampler(choices))
        elif "#" in var:  # use only data from column
            out = sample[var.replace("#", "")]
            # if *_smiles is nan sample from *_name
            if (
                not isinstance(out, str)
                and math.isnan(out)
                and var.find("_smiles") != -1
            ):
                out = sample[var.replace("_smiles", "_name").replace("#", "")]
            # if *_protein_names is nan sample from *_name
            elif (
                not isinstance(out, str)
                and math.isnan(out)
                and var.find("_protein_names") != -1
            ):
                out = sample[var.replace("_protein_names", "_name").replace("#", "")]

        # sampling based on row data and their definiton in the row
        if "|" in out:  # datafield sampling of multiple options
            choices = out.split("|")
            choices = [c for c in choices if (isinstance(c, str) or not math.isnan(c))]
            out = unwrap_list_length_1(self.column_datafield_sampler(choices))
        return out

    def get_sample_dict(self, sample: pd.Series, template: str):
        """Get sample dict from sample row and template string."""
        input_variables = get_input_variables_from_template(template)
        sample_dict = {
            var: self._get_target_from_row(sample, var)
            if "#" in var
            else get_target_from_string(self.meta, var)()
            for var in input_variables
        }
        return sample_dict

    def get_prompt_template_from_template_idx(self, template_idx: int = None) -> str:
        """Get prompt template from template index."""
        if template_idx is None:
            prompt_template = self.rnd_prompt_templates()
        else:
            prompt_template = self.prompt_templates[template_idx]
        return prompt_template

    def sample(self, sample: pd.Series, template_idx: int = None):
        """Sample text template by data from the sample row.
        The text template can be specified by the template index."""
        prompt_template = self.get_prompt_template_from_template_idx(template_idx)
        sample_dict = self.get_sample_dict(sample, prompt_template.template)
        return prompt_template.insert(sample_dict)

    def __getitem__(self, sample_idx: int, template_idx: int = None):
        """Get item from data with sample and template index.
        A random template will be ised if no template index is handed over."""
        sample = self.df.iloc[sample_idx]
        return self.sample(sample, template_idx)

    def apply_sampling(self, template_idx: int = None):
        """Applies the sampling to the entire data frame."""
        self.df["sample"] = self.df.apply(
            lambda sample: self.sample(sample, template_idx), axis=1
        )

    def export(self):
        """Exports the sampled data as separate jsonl files based on the split and benchmarking templates."""
        assert "sample" in self.df.columns, "Run apply_sampling before running export."
        print_data = {
            "split": [],
            "rows": [],
            "path": [],
        }
        for split in self.df.split.unique():
            # subselect for split
            df_out = self.df.copy()
            df_out = df_out[df_out["split"] == split]
            if self.benchmarking_templates:
                df_out[["input", "output"]] = df_out["sample"].str.split(
                    pat="<EOI>", n=1, expand=True
                )
                df_out.drop(
                    [
                        x
                        for x in df_out.columns.tolist()
                        if x not in ["input", "output"]
                    ],
                    axis=1,
                    inplace=True,
                )
            else:
                df_out.drop(
                    [x for x in df_out.columns.tolist() if x != "sample"],
                    axis=1,
                    inplace=True,
                )
                df_out.columns = ["text"]

            # save
            if self.benchmarking_templates:
                output_path = self.path_data_dir + f"/{split}_benchmark.jsonl"
            else:
                output_path = self.path_data_dir + f"/{split}.jsonl"
            with open(output_path, "w") as f:
                f.write(df_out.to_json(orient="records", lines=True, force_ascii=False))

            # stats
            rows_split = len(df_out)
            print_data["split"].append(split)
            print_data["rows"].append(rows_split)
            print_data["path"].append(output_path)

        if len(self.df.split.unique()) > 1:
            print_data["split"].append("total")
            print_data["rows"].append(len(self.df))
            print_data["path"].append("")
        return pd.DataFrame(print_data)

    def apply_sampling_and_export(self, template_idx: int = None):
        """Applies the sampling and exports the data."""
        self.apply_sampling(template_idx)
        df_results = self.export()
        print(df_results.to_string())


if __name__ == "__main__":
    path_base = __file__.replace("text_sampling/text_sampling.py", "")
    path_data_dir = glob.glob(path_base + "tabular/*") + glob.glob(path_base + "kg/*")
    for path in path_data_dir:
        path_meta = path + "/meta.yaml"
        path_data = path + "/data_clean.csv"
        if os.path.isfile(path_meta) and os.path.isfile(path_data):
            meta = load_yaml(path_meta)
            if "templates" in meta:
                print(f"Running sampling for: {path}")
                TemplateSampler(
                    path, benchmarking_templates=False
                ).apply_sampling_and_export()
                if any(["<EOI>" in t for t in meta["templates"]]):
                    TemplateSampler(
                        path, benchmarking_templates=True
                    ).apply_sampling_and_export()
