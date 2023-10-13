from datasets import concatenate_datasets, load_dataset

PUBCHEM_DATASET = "alxfgh/PubChem_Drug_Instruction_Tuning"
CHEMBL_DATASET = "alxfgh/ChEMBL_Drug_Instruction_Tuning"


if __name__ == "__main__":
    # Load the two datasets
    dataset1 = load_dataset(PUBCHEM_DATASET)
    dataset2 = load_dataset(CHEMBL_DATASET)

    # Verify that the datasets have the same schema (i.e., the same fields)
    assert (
        dataset1["train"].features == dataset2["train"].features
    ), "Datasets do not have the same schema"

    # Concatenate the 'train' split of dataset2 to the 'train' split of dataset1
    combined_dataset = concatenate_datasets([dataset1["train"], dataset2["train"]])

    # Define the fractions for train/test/valid split
    train_fraction = 0.8
    test_fraction = 0.1
    # The remaining part will be the validation fraction

    # Generate the train/test/valid splits
    train_test_valid_datasets = combined_dataset.train_test_split(
        test_size=test_fraction, shuffle=True
    )
    train_valid_datasets = train_test_valid_datasets["train"].train_test_split(
        test_size=(1 - train_fraction) / (1 - test_fraction), shuffle=True
    )

    final_datasets = {
        "train": train_valid_datasets["train"],
        "test": train_test_valid_datasets["test"],
        "valid": train_valid_datasets["test"],
    }

    # Add the 'split' column to each dataset
    for split in final_datasets:
        final_datasets[split] = final_datasets[split].add_column(
            "split", [split] * len(final_datasets[split])
        )

    # Concatenate all splits again
    all_datasets = concatenate_datasets(
        [final_datasets[split] for split in final_datasets]
    )
    df = all_datasets.to_pandas()

    df.rename(columns={"Answer": "answ", "Question": "quest"}, inplace=True)

    # Save the combined dataset as a CSV file
    df.to_csv("data_clean.csv", index=False)
