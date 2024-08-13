from chemnlp.data.constants import DEFAULT_SIGNIFICANT_DIGITS
import pandas as pd
import random
import math
from typing import List, Dict, Union, Callable, Optional, Tuple
import re
from string import ascii_lowercase, ascii_uppercase
from chemnlp.data.random_variable import RandomVariable
from functools import partial
from functools import lru_cache
from chemnlp.data_val.model import IdentifierEnum


# ToDo: handle somewhere that the meta contains multiple templates
class TemplateSampler:
    """
    A class for sampling and generating text based on templates and data.

    This class handles the creation of text samples from templates, managing both
    standard variable substitution and multiple-choice question generation. It supports
    various data types and sampling methods, including class-balanced sampling and
    benchmarking templates.

    Attributes:
        df (pd.DataFrame): The dataset used for sampling.
        meta (Dict): Metadata about the dataset, including identifiers and targets.
        config (Dict): Configuration parameters for the sampler.
        column_datafield_sampler (Callable): A function for sampling from multiple options.

    Examples:
            >>> config = {
            ...     'DEFAULT_SIGNIFICANT_DIGITS': 3,
            ...     'multiple_choice_rnd_symbols': ["", ".", ".)", ")", ":", "()", "[]"],
            ...     'multiple_choice_benchmarking_templates': False,
            ...     'multiple_choice_benchmarking_format': None
            ... }
            >>> sampler = TemplateSampler(df, meta, config)
            >>> template = "The molecule with SMILES {SMILES#} has a {property#} of {value#}."
            >>> result = sampler.sample(df.iloc[0], template)
            >>> print(result)
            The molecule with SMILES CC(=O)OC1=CC=CC=C1C(=O)O has a solubility of 3.142.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        meta: Dict,
        config: Dict,
        column_datafield_sampler: Optional[Callable] = None,
    ):
        self.df_orig = df
        self.df = df
        self.meta = meta
        self.config = config
        self.column_datafield_sampler = column_datafield_sampler or (
            lambda x: random.sample(x, k=1)
        )
        self.class_balanced = False
        self.balance_column = None
        self.wrap_identifiers = config.get("wrap_identifiers", False)

    def _wrap_identifier(self, identifier: str, value: str) -> str:
        """Wrap the identifier value with tags if wrap_identifiers is enabled."""

        if not self.wrap_identifiers:
            return value

        identifier_type = next(
            (
                item["type"]
                for item in self.meta["identifiers"]
                if item["id"] == identifier
            ),
            None,
        )

        try:
            identifier_type = IdentifierEnum(identifier_type)
        except ValueError:
            identifier_type = None

        if identifier_type:
            return f"[BEGIN_{identifier_type}]{value}[END_{identifier_type}]"
        return value

    def _balance_classes(self, column: str) -> pd.DataFrame:
        """
        Create a class-balanced version of the dataset.

        Args:
            column (str): The column to use for balancing.

        Returns:
            pd.DataFrame: A new dataframe with balanced classes.
        """
        value_counts = self.df_orig[column].value_counts()
        min_count = value_counts.min()
        balanced_dfs = []

        for value in value_counts.index:
            class_df = self.df_orig[self.df_orig[column] == value]
            if len(class_df) > min_count:
                class_df = class_df.sample(min_count)
            balanced_dfs.append(class_df)

        return pd.concat(balanced_dfs, ignore_index=True)

    def enable_class_balancing(self, column: str):
        """
        Enable class-balanced sampling.

        Args:
            column (str): The column to use for balancing.
        """
        self.class_balanced = True
        self.balance_column = column
        self.df = self._balance_classes(column)

    def disable_class_balancing(self):
        """
        Disable class-balanced sampling and revert to the original dataset.
        """
        self.class_balanced = False
        self.balance_column = None
        self.df = self.df_orig

    def _get_target_from_row(self, sample: pd.Series, var: str) -> str:
        """
        Extract and process a target value from a sample row based on a variable string.

        This method handles various formats of the variable string to extract and process
        data from the sample row. It supports multiple text string sampling, recoding,
        multiple column selection, and special case handling for NaN values.

        The method also processes the extracted value based on its data type (continuous or not)
        and handles cases where the extracted value itself contains multiple options.

        Args:
            sample (pd.Series): A row from the dataset.
            var (str): A string specifying how to extract the target value. This can include
                        special characters like '#', '!', '|', and '&' for different behaviors.

        Returns:
            str: The extracted and processed target value.

        Raises:
            ValueError: If a continuous value is not a number (float or int).

        Note:
            - The behavior changes based on the format of the 'var' string:
                - '#', '!', '|': Treats as synonm options. One is randomly selected.
                - '#', '&': Treats as recoding information.
                - '#', '|': Treats as multiple column selection.
                - Only '#': Simple column value retrieval.
            - Special handling is included for NaN values in certain column types.
            - The method rounds continuous values to a specified number of significant digits.
            - If the final value contains '|', it's split and a random option is chosen.
        """
        if ("#" in var) and ("!" in var) and ("|" in var):
            choices = var.replace("#", "").replace("!", "").split("|")
            return self.column_datafield_sampler(choices)[0]

        elif ("#" in var) and ("&" in var):
            var, choices = var.split("#")
            choices = choices.split("&")
            print("var and choices and sample", var, choices, sample)
            choice = choices[sample[var]]
            return "" if choice == "NULL" else choice

        elif ("#" in var) and ("|" in var):
            var = var.replace("#", "")
            columns = var.split("|")
            var = self.column_datafield_sampler(columns)[0]
            out = sample[var]

        elif "#" in var:
            var = var.replace("#", "")
            out = sample[var]
            if not isinstance(out, str) and math.isnan(out):
                if "_smiles" in var:
                    out = sample[var.replace("_smiles", "_name")]
                elif "_protein_names" in var:
                    out = sample[var.replace("_protein_names", "_name")]

        var_dict = next(
            x for x in self.meta["identifiers"] + self.meta["targets"] if x["id"] == var
        )
        if var_dict["type"] == "continuous":
            if not isinstance(out, (float, int)):
                raise ValueError(f"out is not a number (int or float): {out}")
            significant_digits = var_dict.get(
                "significant_digits",
                self.config.get(
                    "DEFAULT_SIGNIFICANT_DIGITS", DEFAULT_SIGNIFICANT_DIGITS
                ),
            )
            out = f"{round(out, significant_digits):.{significant_digits}f}"
        else:
            out = str(out)

        if "|" in out:
            choices = [
                c for c in out.split("|") if isinstance(c, str) or not math.isnan(c)
            ]
            out = self.column_datafield_sampler(choices)[0]

        return out

    def get_sample_dict(self, sample: pd.Series, template: str) -> Dict[str, str]:
        """
        Extract and process all target values from a sample row based on a template.
        """
        input_variables = self._get_input_variables_from_template(template)
        sample_dict = {}

        if any("%" in x for x in input_variables):
            sample_dict.update(self._handle_multiple_choice(sample, input_variables))

        for var in input_variables:
            if "#" in var:
                sample_dict[var] = self._get_target_from_row(sample, var)
            elif "%" not in var:
                sample_dict[var] = self._get_target_from_string(var)()

        return sample_dict

    def _get_symbols_from_multiple_choice_enum(self, enum_str: str) -> List[str]:
        _, choice_count, symbol = enum_str.split("%")[1:]
        if "-" in choice_count:
            min_count, max_count = map(int, choice_count.split("-"))
            count = random.randint(min_count, max_count)
        else:
            count = int(choice_count)

        if "a" in symbol:
            return list(ascii_lowercase[:count])
        elif "A" in symbol:
            return list(ascii_uppercase[:count])
        elif "1" in symbol:
            return [str(i) for i in range(1, count + 1)]

    def _format_enum_string(self, symbols: List[str]) -> str:
        """
        Format a list of symbols into a string representation for multiple-choice questions.

        This method takes a list of symbols (e.g., ['a', 'b', 'c']) and formats them
        into a string like "a, b, or c" for use in multiple-choice question prompts.

        Args:
            symbols (List[str]): A list of symbols representing multiple-choice options.

        Returns:
            str: A formatted string of the symbols.

        Examples:
            ['a', 'b']       -> "a or b"
            ['a', 'b', 'c']  -> "a, b, or c"
            ['1', '2', '3', '4'] -> "1, 2, 3, or 4"
        """
        if len(symbols) == 0:
            return ""
        elif len(symbols) == 1:
            return symbols[0]
        elif len(symbols) == 2:
            return f"{symbols[0]} or {symbols[1]}"
        else:
            return ", ".join(symbols[:-1]) + f", or {symbols[-1]}"

    def _handle_multiple_choice(
        self, sample: pd.Series, input_variables: List[str]
    ) -> Dict[str, Union[str, List[str]]]:
        multiple_choice_dict = {}

        # get multiple_choice_enum
        multiple_choice_enum_idx = [
            i
            for i, x in enumerate(input_variables)
            if x.startswith("%multiple_choice_enum")
        ]
        assert len(multiple_choice_enum_idx) == 1
        multiple_choice_enum_idx = multiple_choice_enum_idx[0]
        multiple_choice_enum = input_variables[multiple_choice_enum_idx]

        # get multiple_choice_var
        multiple_choice_var_idx = [
            i for i, x in enumerate(input_variables) if x.endswith("%")
        ]
        assert len(multiple_choice_var_idx) == 1
        multiple_choice_var_idx = multiple_choice_var_idx[0]
        multiple_choice_input = input_variables[multiple_choice_var_idx]

        if multiple_choice_input.count("%") > 1:
            multiple_choice_var, multiple_choice_indicator, _ = (
                multiple_choice_input.split("%")
            )
        else:
            multiple_choice_var, multiple_choice_indicator = (
                multiple_choice_input.split("%")
            )
            multiple_choice_indicator = (
                ""  # multiple_choice_indicator is here an empty string
            )

        symbols = self._get_symbols_from_multiple_choice_enum(multiple_choice_enum)

        # get all and correct choices incl. index
        correct_choice = self._get_target_from_row(sample, multiple_choice_var + "#")

        if multiple_choice_indicator == "":
            multiple_choices, correct_choice_idx = self._get_choices_without_indicator(
                multiple_choice_var, symbols, correct_choice
            )
        else:
            multiple_choices, correct_choice_idx = self._get_choices_with_indicator(
                sample,
                multiple_choice_var,
                multiple_choice_indicator,
                symbols,
                correct_choice,
            )

        multiple_choice_dict[multiple_choice_enum] = self._format_enum_string(symbols)
        multiple_choice_dict[multiple_choice_input] = self._format_choices(
            symbols, multiple_choices
        )
        multiple_choice_dict["%multiple_choice_result"] = self._format_result(
            symbols, correct_choice_idx
        )
        multiple_choice_dict["%multiple_choice_symbols"] = symbols
        multiple_choice_dict["%multiple_choice_result_idx"] = correct_choice_idx

        return multiple_choice_dict

    def _get_choices_without_indicator(
        self, multiple_choice_var: str, symbols: List[str], correct_choice: str
    ) -> Tuple[List[str], int]:
        cutoff_full_unique = 100
        all_choices = self.df[multiple_choice_var].unique()
        if len(all_choices) > cutoff_full_unique:
            all_choices = (
                self.df[multiple_choice_var].sample(cutoff_full_unique).unique()
            )
        all_choices = sorted([str(x) for x in all_choices])

        if all_choices == ["0", "1"]:
            all_choices = ["False", "True"]
            correct_choice = all_choices[int(correct_choice)]

        multiple_choices = random.sample(all_choices, k=len(symbols))
        if correct_choice not in multiple_choices:
            multiple_choices = multiple_choices[:-1] + [correct_choice]
            random.shuffle(multiple_choices)

        correct_choice_idx = multiple_choices.index(correct_choice)
        return multiple_choices, correct_choice_idx

    def _get_choices_with_indicator(
        self,
        sample: pd.Series,
        multiple_choice_var: str,
        multiple_choice_indicator: str,
        symbols: List[str],
        correct_choice: str,
    ) -> Tuple[List[str], List[int]]:
        correct_choice_indicator = self._get_target_from_row(
            sample, multiple_choice_indicator + "#"
        )
        df_sample = self.df.sample(len(symbols) - 1)[
            [multiple_choice_var, multiple_choice_indicator]
        ]

        multiple_choices = df_sample[multiple_choice_var].astype(str).tolist() + [
            correct_choice
        ]
        multiple_choices_indicators = df_sample[multiple_choice_indicator].astype(
            str
        ).tolist() + [correct_choice_indicator]

        multiple_choices_combined = list(
            zip(multiple_choices, multiple_choices_indicators)
        )
        random.shuffle(multiple_choices_combined)
        multiple_choices, multiple_choices_indicators = zip(*multiple_choices_combined)

        correct_choice_idx = [
            i
            for i, (choice, indicator) in enumerate(
                zip(multiple_choices, multiple_choices_indicators)
            )
            if indicator == correct_choice_indicator
        ]

        return list(multiple_choices), correct_choice_idx

    def _format_choices(self, symbols: List[str], choices: List[str]) -> str:
        rnd_symbol = self._get_random_symbol()
        rnd_symbol_prefix, rnd_symbol_suffix = self._get_symbol_affixes(rnd_symbol)

        return "\n".join(
            [
                f"{rnd_symbol_prefix}{s}{rnd_symbol_suffix} {c}"
                for s, c in zip(symbols, choices)
            ]
        )

    def _format_result(
        self, symbols: List[str], correct_choice_idx: Union[int, List[int]]
    ) -> str:
        if isinstance(correct_choice_idx, list):
            return ", ".join([symbols[i] for i in correct_choice_idx])
        else:
            return symbols[correct_choice_idx]

    def _get_random_symbol(self) -> str:
        if (
            self.config.get("multiple_choice_benchmarking_templates")
            and self.config.get("multiple_choice_benchmarking_format") is not None
        ):
            if len(self.config["multiple_choice_rnd_symbols"]) > 1:
                return self.config["multiple_choice_rnd_symbols"][
                    self.config["multiple_choice_benchmarking_format"]
                ]
            else:
                return self.config["multiple_choice_rnd_symbols"][0]
        else:
            return random.choice(self.config["multiple_choice_rnd_symbols"])

    def _get_symbol_affixes(self, symbol: str) -> Tuple[str, str]:
        if symbol in ["()", "[]"]:
            return symbol[0], symbol[1]
        else:
            return "", symbol

    def _get_input_variables_from_template(self, template: str) -> List[str]:
        return re.findall(r"\{([^}]+)\}", template)

    @lru_cache(maxsize=None)
    def _get_random_text_identifiers_and_targets(self) -> dict:
        """Cached version of get_random_text_identifiers_and_targets"""
        rnd_texts = {}
        for e in self.meta["identifiers"] + self.meta["targets"]:
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

            if "description" in e:
                rnd_texts[e["id"]]["description"] = partial(
                    lambda x: x, e["description"]
                )

            if "units" in e:
                rnd_texts[e["id"]]["units"] = partial(lambda x: x, e["units"])

        return rnd_texts

    def _get_target_from_string(self, var: str) -> str:
        """
        Retrieve a target value from the meta information based on a string key.

        This method navigates through the nested structure of the meta dictionary
        to find the appropriate value.

        Args:
            var (str): A string key representing the path to the target value
                    in the meta dictionary, with levels separated by '__'.

        Returns:
            str: The target value.

        Raises:
            KeyError: If the specified path doesn't exist in the meta dictionary.

        Example:
            If var is "SMILES__names__noun", it will look for
            self.meta["SMILES"]["names"]["noun"] and return a RandomVariable if it's a list.
        """
        keys = var.split("__")

        def get_with_nested_keys(d: dict, keys: list) -> Union[str, Callable]:
            t = d
            for k in keys:
                if k not in t:
                    raise KeyError(f"Key '{k}' not found in nested dictionary.")
                t = t[k]
            return t

        if len(keys) == 1 and keys[0] in self.meta:
            return self.meta[keys[0]]
        elif keys[0] in [
            x["id"] for x in self.meta["identifiers"] + self.meta["targets"]
        ]:
            rnd_texts = self._get_random_text_identifiers_and_targets()
            return get_with_nested_keys(rnd_texts, keys)
        else:
            raise KeyError(f"Unable to find key '{var}' in meta information.")

    def sample(self, sample: pd.Series, template: str) -> str:
        """
        Generate a text sample based on a template and a data sample.

        If no sample is provided, a random sample is chosen from the current dataset
        (which may be class-balanced if enabled).

        Args:
            sample (Optional[pd.Series]): A row from the dataset. If None, a random sample is chosen.
            template (str): The template string to be filled.

        Returns:
            str: The completed text sample with all variables replaced by their values.
        """
        if sample is None:
            sample = self.df.sample(1).iloc[0]
        sample_dict = self.get_sample_dict(sample, template)
        return self._fill_template(template, sample_dict)

    def _fill_template(
        self, template: str, sample_dict: Dict[str, Union[str, List[str]]]
    ) -> str:
        for key, value in sample_dict.items():
            if isinstance(value, list):
                value = "\n".join(value)
            if "#" in key:  # This indicates it's an identifier
                identifier = key.replace("#", "")
                value = self._wrap_identifier(identifier, str(value))
            template = template.replace("{" + key + "}", str(value))
        return template
