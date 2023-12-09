import pandas as pd
import yaml


def get_and_transform_data():
    # load data
    df = pd.read_csv(
        "https://tatonettilab.org/resources/nsides/OFFSIDES.csv.gz",
        compression="gzip",
        on_bad_lines="skip",
        low_memory=False,
    )

    # check if fields are the same
    expected_columns = [
        "drug_rxnorn_id",
        "drug_concept_name",
        "condition_meddra_id",
        "condition_concept_name",
        "A",
        "B",
        "C",
        "D",
        "PRR",
        "PRR_error",
        "mean_reporting_frequency",
    ]

    assert df.columns.tolist() == expected_columns

    # drop columns A, B, C, D
    df.drop(columns=["A", "B", "C", "D"], inplace=True)
    # remove duplicates
    df.drop_duplicates(inplace=True)
    # check duplicates
    assert not df.duplicated().sum(), "Found duplicate rows in the dataframe"
    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "offsides",
        "description": """OffSIDES is a database of individual drug side effect
signals mined from the FDA's Adverse Event Reporting System. The
innovation of OffSIDES is that a propensity score matching (PSM) model
is used to identify control drugs and produce better PRR estimates. In
OffSIDES we focus on drug safety signals that are not already
established by being listed on the structured product label - hence
they are off-label drug side effects.""",
        "targets": [
            {
                "id": "PRR",
                "description": "proportional reporting ratio",
                "type": "continuous",
                "names": ["proportional reporting ratio"],
            },
            {
                "id": "PRR_error",
                "description": "standard error of the PRR estimate",
                "type": "continuous",
                "sample": False,
                "names": ["standard error of the proportional reporting ratio error"],
            },
            {
                "id": "mean_reporting_frequency",
                "description": "mean reporting frequency for the drug",
                "type": "continuous",
                "names": ["mean reporting frequency"],
            },
        ],
        "identifier": [
            {
                "id": "drug_concept_name",
                "description": "RxNorm name string for the drug",
                "type": "categorical",
            },
            {
                "id": "condition_concept_name",
                "description": "MedDRA identifier for the side effect",
                "type": "categorical",
            },
        ],
        "license": "CC BY 4.0",
        "links": [
            {
                "url": "https://tatonettilab.org/resources/nsides/",
                "description": "data source",
            },
            {"url": "https://nsides.io/", "description": "database website"},
        ],
        "num_points": len(df),
        "bibtex": """@article{Tatonetti2012,
author = {Tatonetti, Nicholas P. and Ye, Peter P. and Daneshjou, Roxana and Altman, Russ B.},
title = {Data-driven prediction of drug effects and interactions},
journal = {Sci Transl Med},
volume = {4},
number = {125},
pages = {125ra31},
year = {2012},
doi = {10.1126/scitranslmed.3003377},
pmid = {22422992},
pmcid = {PMC3382018}
}""",
    }

    def str_presenter(dumper, data):
        """configures yaml for dumping multiline strings
        Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
        """
        if data.count("\n") > 0:  # check for multiline string
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(
        str, str_presenter
    )  # to use with safe_dum
    fn_meta = "meta.yaml"
    with open(fn_meta, "w") as f:
        yaml.dump(meta, f, sort_keys=False)

    print(f"Finished processing {meta['name']} dataset!")


if __name__ == "__main__":
    get_and_transform_data()
