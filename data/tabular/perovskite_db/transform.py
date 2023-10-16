import ast

import pandas as pd
from datasets import load_dataset


def oxford_comma_join(list_of_str):
    if len(list_of_str) == 1:
        return list_of_str[0]
    elif len(list_of_str) == 2:
        return " and ".join(list_of_str)
    else:
        return ", ".join(list_of_str[:-1]) + ", and " + list_of_str[-1]


def preprocess():
    df = load_dataset("kjappelbaum/pervoskite_db", delimiter="|")["train"].to_pandas()
    df.dropna(
        subset=[
            "device_stack",
            "pce",
            "ff",
            "jsc",
            "voc",
            "reduced_formulas",
            "descriptive_formulas",
            "iupac_formulas",
            "bandgap",
        ],
        inplace=True,
    )
    device_stack_strings = []

    df["pce"] = df["pce"].round(2)
    df["ff"] = df["ff"].round(2)
    df["jsc"] = df["jsc"].round(2)
    df["voc"] = df["voc"].round(2)
    df["bandgap"] = df["bandgap"].round(2)

    for i, row in df.iterrows():
        device_stack = ast.literal_eval(row["device_stack"])
        device_stack_string = oxford_comma_join(device_stack)
        absorber = row["descriptive_formulas"]
        device_stack_string = device_stack_string.replace("Perovskite", absorber)
        device_stack_strings.append(device_stack_string)

    df["device_stack_string"] = device_stack_strings
    df[
        [
            "pce",
            "ff",
            "jsc",
            "voc",
            "bandgap",
            "reduced_formulas",
            "descriptive_formulas",
            "iupac_formulas",
            "device_stack_string",
        ]
    ].to_csv("data_clean.csv", index=False)
    print(len(df))


if __name__ == "__main__":
    preprocess()
