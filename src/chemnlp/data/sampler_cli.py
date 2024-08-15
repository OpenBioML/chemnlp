import os
import fire
import pandas as pd
import random
import warnings
from typing import Optional, List
from chemnlp.data.sampler import TemplateSampler
from chemnlp.data.utils import load_yaml
from chemnlp.data.constants import (
    STANDARD_TABULAR_TEXT_TEMPLATES,
    EXCLUDE_FROM_STANDARD_TABULAR_TEXT_TEMPLATES,
    DEFAULT_SIGNIFICANT_DIGITS,
)
from loguru import logger
from pathlib import Path


def determine_balance_column(meta: dict, template: str) -> Optional[str]:
    """
    Determine which column to use for class balancing based on the template and metadata.
    """
    target_ids = [target["id"] for target in meta["targets"]]
    template_targets = [var.split("#")[0] for var in template.split("{") if "#" in var]

    matching_targets = [target for target in template_targets if target in target_ids]

    if not matching_targets:
        return None
    elif len(matching_targets) == 1:
        return matching_targets[0]
    else:
        chosen_target = random.choice(matching_targets)
        warnings.warn(
            f"Multiple targets found in template. Randomly chose '{chosen_target}' for balancing."
        )
        return chosen_target


def process_dataset(
    data_dir: str,
    output_dir: str,
    chunksize: int = 1_000_000,
    class_balanced: bool = False,
    benchmarking: bool = False,
    additional_templates: Optional[List[str]] = None,
    use_standard_templates: bool = True,
    wrap_identifiers: bool = False,
):
    """
    Process a dataset using TemplateSampler.

    Args:
        data_dir (str): Path to directory containing meta.yaml and data_clean.csv
        output_dir (str): Path to directory for output files
        chunksize (int): Number of rows to process at a time
        class_balanced (bool): Whether to use class balancing
        benchmarking (bool): Whether to use benchmarking templates
        additional_templates (List[str], optional): Additional templates to use
        use_standard_templates (bool): Whether to use standard tabular text templates
        wrap_identifiers (bool): Whether to wrap identifiers in templates
    """
    meta_path = os.path.join(data_dir, "meta.yaml")
    data_path = os.path.join(data_dir, "data_clean.csv")

    meta = load_yaml(meta_path)

    # Add standard templates if applicable
    if use_standard_templates and "/tabular/" in data_dir:
        dataset_name = os.path.basename(data_dir.strip("/"))
        logger.info(f"Adding standard templates for dataset '{dataset_name}'")
        if dataset_name not in EXCLUDE_FROM_STANDARD_TABULAR_TEXT_TEMPLATES:
            logger.info(f"Adding standard templates for dataset since it is not excluded '{dataset_name}'")
            if any(identifier["id"] == "SMILES" for identifier in meta["identifiers"]):
                if len(meta["targets"]) == 1:
                    target_id = meta["targets"][0]["id"]
                    standard_templates = [
                        template.replace("TARGET", target_id)
                        for template in STANDARD_TABULAR_TEXT_TEMPLATES
                    ]
                    if "templates" in meta:
                        meta["templates"] += standard_templates
                    else:
                        meta["templates"] = standard_templates

    if additional_templates:
        if "templates" in meta:
            meta["templates"] += additional_templates
        else:
            meta["templates"] = additional_templates

    multiple_choice_rnd_symbols = ["", ".", ".)", ")", ":", "()", "[]"]

    config = {
        "DEFAULT_SIGNIFICANT_DIGITS": DEFAULT_SIGNIFICANT_DIGITS,
        "multiple_choice_rnd_symbols": multiple_choice_rnd_symbols,
        "multiple_choice_benchmarking_templates": True,
        "multiple_choice_benchmarking_format": None,
        "wrap_identifiers": wrap_identifiers,
        "benchmarking_templates": benchmarking,
        "excluded_from_wrapping": ["Other"],
    }

    templates = meta["templates"]
    if benchmarking:
        templates = [t for t in templates if "<EOI>" in t]
    else:
        templates = [t for t in templates if "<EOI>" not in t]
    output_dir_ = os.path.join(Path(output_dir), os.path.basename(data_dir.strip("/")))
    os.makedirs(output_dir_, exist_ok=True)
    for chunk_idx, df_chunk in enumerate(
        pd.read_csv(data_path, chunksize=chunksize, low_memory=False)
    ):
        chunk_output_dir = os.path.join(output_dir_, f"chunk_{chunk_idx}")
        logger.debug(f"Processing chunk {chunk_idx} to {chunk_output_dir}")
        os.makedirs(chunk_output_dir, exist_ok=True)

        sampler = TemplateSampler(df_chunk, meta, config, data_dir)

        for template_idx, template in enumerate(templates):
            print(
                f"\nProcessing chunk {chunk_idx}, template {template_idx}:\n{template}"
            )

            # Determine balance column
            balance_column = (
                determine_balance_column(meta, template) if class_balanced else None
            )

            # Enable class balancing if needed
            if balance_column:
                sampler.enable_class_balancing(balance_column)
                print(f"Enabled class balancing on column: {balance_column}")

            # Export step
            template_output_dir = os.path.join(
                chunk_output_dir, f"template_{template_idx}"
            )
            result_df = sampler.export(template_output_dir, template)

            print(f"Exported samples to {template_output_dir}")
            print(result_df)

            if balance_column:
                sampler.disable_class_balancing()


def main(
    data_dir: str,
    output_dir: str,
    chunksize: int = 1_000_000,
    class_balanced: bool = False,
    benchmarking: bool = False,
    additional_templates: Optional[List[str]] = None,
    use_standard_templates: bool = True,
    wrap_identifiers: bool = False,
):
    """
    Main function to run the sampler CLI.

    Args:
        data_dir (str): Path to directory containing meta.yaml and data_clean.csv
        output_dir (str): Path to directory for output files
        chunksize (int): Number of rows to process at a time
        class_balanced (bool): Whether to use class balancing
        benchmarking (bool): Whether to use benchmarking templates
        additional_templates (List[str], optional): Additional templates to use
        use_standard_templates (bool): Whether to use standard tabular text templates
        wrap_identifiers (bool): Whether to wrap identifiers in templates
    """
    process_dataset(
        data_dir,
        output_dir,
        chunksize,
        class_balanced,
        benchmarking,
        additional_templates,
        use_standard_templates,
        wrap_identifiers,
    )


def cli():
    fire.Fire(main)


if __name__ == "__main__":
    fire.Fire(main)
