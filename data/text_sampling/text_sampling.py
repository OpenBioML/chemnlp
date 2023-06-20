import glob
import math
import os.path
import random
import re
from functools import partial

import pandas as pd
import yaml


def unwrap_list_length_1(sample):
    """Unwraps lists of length 1."""
    if isinstance(sample, list):
        assert len(sample) == 1
        return sample[0]
    else:
        raise NotImplementedError()


class RandomVariable:
    def __init__(self, name, data, sampler=None):
        self.name = name
        self.data = data
        self.sampler = partial(random.sample, k=1) if sampler is None else sampler

    def __repr__(self):
        return f"RandomVariable: {self.name}, {self.data}, {self.sampler}"

    def __call__(self):
        return unwrap_list_length_1(self.sampler(self.data))


def load_yaml(path):
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def get_input_variables_from_template(template: str):
    return [x for x in re.findall(r"\{([^}]+)\}", template)]


class PromptTemplate:
    def __init__(self, template, input_variables=None):
        self.template = template
        if input_variables is None:
            self.input_variables = self.get_input_variables()

    def get_input_variables(self):
        return get_input_variables_from_template(self.template)

    def __repr__(self):
        return f"PromptTemplate: {self.template}"

    def insert(self, data):
        # check that we got the data for the input_variables, more doesn't matter
        assert all(x in data.keys() for x in self.input_variables)
        template = self.template
        for k in data:
            template = template.replace("{" + k + "}", data[k])
        return template


def get_random_text_identifiers_and_targets(meta):
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


def get_target_from_string(meta, string):
    keys = string.split("__")

    def get_with_nested_keys(d, keys):
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
    def __init__(
        self,
        path_data_dir,
        template_sampler=None,
        column_datafield_sampler=None,
        benchmarking_templates=False,
    ):
        # paths
        self.path_data_dir = path_data_dir
        self.path_data_meta = self.path_data_dir + "/meta.yaml"
        self.path_data_csv = self.path_data_dir + "/data_clean.csv"

        # meta from yaml
        self.meta = load_yaml(self.path_data_meta)

        # dataframe from csv
        df = pd.read_csv(self.path_data_csv)

        def check_targets_and_identifiers(meta, df):
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

    def _get_target_from_row(self, sample, var):
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

    def get_sample_dict(self, sample, template):
        input_variables = get_input_variables_from_template(template)
        sample_dict = {
            var: self._get_target_from_row(sample, var)
            if "#" in var
            else get_target_from_string(self.meta, var)()
            for var in input_variables
        }
        return sample_dict

    def get_prompt_template_from_template_idx(self, template_idx=None):
        if template_idx is None:
            prompt_template = self.rnd_prompt_templates()
        else:
            prompt_template = self.prompt_templates[template_idx]
        return prompt_template

    def sample(self, sample, template_idx=None):
        prompt_template = self.get_prompt_template_from_template_idx(template_idx)
        sample_dict = self.get_sample_dict(sample, prompt_template.template)
        return prompt_template.insert(sample_dict)

    def __getitem__(self, sample_idx, template_idx=None):
        sample = self.df.iloc[sample_idx]
        return self.sample(sample, template_idx)

    def apply_sampling(self, template_idx=None):
        self.df["sample"] = self.df.apply(
            lambda sample: self.sample(sample, template_idx), axis=1
        )

    def export(self):
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

        print_data["split"].append("total")
        print_data["rows"].append(len(self.df))
        print_data["path"].append("")
        return pd.DataFrame(print_data)

    def apply_and_export(self, template_idx=None):
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
                TemplateSampler(path, benchmarking_templates=False).apply_and_export()
                if any(["<EOI>" in t for t in meta["templates"]]):
                    TemplateSampler(
                        path, benchmarking_templates=True
                    ).apply_and_export()
