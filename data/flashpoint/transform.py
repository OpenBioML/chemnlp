import os
from hashlib import md5
from typing import Dict

import numpy as np
import pandas as pd
import requests
import yaml

# md5 from the file downloaded on Feb 25, 2023
MD5_SUM = "71e3f7e37e96a9381eb9b91f8c18a025"


def get_and_transform_data(
    orig_data_path: str = "data_orig.csv",
    output_data_path: str = "data_clean.csv",
    md5_sum: str = MD5_SUM,
):
    """Downloads and processes the data and saves it as a csv in output_data_path.
    Args:
        orig_data_path: path to save the downloaded original data file
        output_data_path: path to save the processed data file
    """
    # Read meta.yaml
    with open("meta.yaml", "r") as f:
        metadata = yaml.safe_load(f)

    dataset_name = metadata["name"]

    # The data from the metadata's url is downloaded to orig_data_path
    _download_orig_data(
        metadata=metadata, orig_data_path=orig_data_path, md5_sum=md5_sum
    )
    print(
        f"The preprocessed {dataset_name} dataset was saved to "
        + f"{os.path.abspath(orig_data_path)}."
    )

    print(f"Processing {dataset_name} dataset...")

    orig_df = pd.read_csv(orig_data_path)
    print(f"\t The original dataset has {len(orig_df)} datapoints.")

    fields_orig = orig_df.columns.tolist()
    assert fields_orig == [
        "compound",  # e.g., "bromocyclopentane"
        "flashpoint",
        "pure substance",  # presumably binary, but all are 1
        "smiles",
        "source",  # source of data, e.g., "pubchem", "carroll11"
        "is_silicon",  # binary, 0 or 1
        "is_metallic",  # binary
        "is_tin",  # binary
        "is_acid",  # binary
        "data type",  # e.g., "Non-DIPPR data", "test data"
    ]

    # Partition the df into two, one whose smiles column values are unique
    # and another whose smiles column values are duplicated
    unique_smiles_subdf, duplicated_smiles_subdf = _get_unique_smiles_subdf(orig_df)
    print(
        f"\t Processed {len(unique_smiles_subdf)} datapoints with unique smiles "
        + "in the original dataset."
    )

    # From the duplicated smiles dataframe, get a smaller dataframe whose smiles
    # values are unique. This smaller dataframe contains the mean of the values
    # for each smiles, after filtering for quality.
    extracted_duplicated_smiles_df = _extract_values_from_duplicate_smiles(
        duplicated_smiles_subdf
    )

    processed_df = pd.concat([unique_smiles_subdf, extracted_duplicated_smiles_df])
    assert len(processed_df) == len(set(processed_df.smiles))
    assert len(processed_df) == metadata["num_points"], (
        "Processed Data does not contain the expected number of points! "
        + f"Expected number={metadata['num_points']}, actual number={len(processed_df)}"
    )

    # These are the only output columns
    cols_to_write = ["smiles", "flashpoint"]

    processed_df = processed_df[cols_to_write]
    processed_df.columns = ["SMILES", "flashpoint"]

    processed_df.to_csv(output_data_path, index=False)
    print(
        f"Finished processing {dataset_name} dataset! ({len(processed_df)} datapoints)"
    )
    print(f"The processed dataset was saved to {os.path.abspath(output_data_path)}.")


def _download_orig_data(metadata: Dict, orig_data_path: str, md5_sum: str):
    """Downloads the data, verifies its md5 hash, and saves it in orig_data_path.
    Args:
        metadata: the metadata in a python dictionary
        orig_data_path: path to save the downloaded original data file
    """
    data_url = metadata["links"][0]["url"]

    # Download data
    response = requests.get(data_url)
    orig_data = response.content

    # Check that the downloaded file is as expected
    assert (
        md5(orig_data).hexdigest() == md5_sum
    ), "Downloaded file does not have the expected checksum!"

    # Save the downloaded data
    with open(orig_data_path, "wb") as f:
        f.write(orig_data)


def _get_unique_smiles_subdf(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Given a dataframe, partition into and return two sub-dataframes, one whose
    smiles column consists of unique values, and another which has duplicated smiles
    values.
    Args:
        df: input dataframe with a smiles column
    Returns:
        unique_smiles_subdf: sub-dataframe whose smiles column values are unique
        duplicate_smiles_subdf: sub-dataframe whose smiles values all have duplicates
    """
    smiles_counts = dict(df.smiles.value_counts())
    unique_smiles = [smiles for smiles in smiles_counts if smiles_counts[smiles] == 1]
    unique_smiles_subdf = df[df["smiles"].isin(unique_smiles)]
    duplicate_smiles_subdf = df[~df["smiles"].isin(unique_smiles)]
    return unique_smiles_subdf, duplicate_smiles_subdf


def _extract_values_from_duplicate_smiles(
    df: pd.DataFrame,
    sigma_threshold: float = 5.0,
) -> pd.DataFrame:
    """Given a dataframe that has duplicated smiles, produces a dataframe whose smiles
    column is unique. The extracted flashpoint value will be the mean of all the
    flashpoint values with the same smiles provided the standard deviation is at most
    sigma_threshold (default 5).
    Note all dataframes below have the same columns as the original data.
    Args:
        df: a dataframe whose smiles values all have duplicates
        sigma_threshold: the threshold for standard deviation to throw out datapoints
    Returns:
        extracted_df: a dataframe with the same columns whose smiles are unique
    """
    rows = []

    std_skipped_smiles = 0
    std_skipped_dps = 0

    processed_smiles = 0
    processed_dps = 0

    for smiles in set(df.smiles):
        smiles_subdf = df[df.smiles == smiles]
        assert (
            len(smiles_subdf) >= 2
        ), "Every smiles in this dataframe should be duplicated!"

        mu, sigma = np.mean(smiles_subdf.flashpoint), np.std(smiles_subdf.flashpoint)
        if sigma > sigma_threshold:
            # If the standard deviation of flashpoint values corresponding to a smiles
            # string is less than sigma_threshold, we keep the mean of these values as
            # the 'true' value. Otherwise, we discard the data points. This is what
            # the Sun et al. paper does with sigma_threshold = 5 (the default).
            std_skipped_smiles += 1
            std_skipped_dps += len(smiles_subdf)
        else:
            processed_smiles += 1
            processed_dps += len(smiles_subdf)
            new_row = smiles_subdf[0:1].copy()
            new_row.flashpoint = mu
            rows.append(new_row)
    print(
        f"\t Skipped {std_skipped_smiles} duplicate smiles with high standard "
        + f"deviation corresponding to {std_skipped_dps} datapoints"
    )
    print(
        f"\t Processed {processed_smiles} duplicate smiles corresponding to "
        + f"{processed_dps} datapoints (collapsed to {processed_smiles} datapoints)."
    )
    extracted_df = pd.concat(rows)
    return extracted_df


if __name__ == "__main__":
    get_and_transform_data()
