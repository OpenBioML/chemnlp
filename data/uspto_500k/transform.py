import pandas as pd
import yaml
from tdc.single_pred import Yields


def get_and_transform_data():
    # get raw data
    data = Yields(name="USPTO_Yields")
    splits = data.get_split()
    df_train = splits["train"]
    df_valid = splits["valid"]
    df_test = splits["test"]
    df_train["split"] = "train"
    df_valid["split"] = "valid"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_valid, df_test], axis=0)

    df["catalyst"] = df.Reaction.apply(lambda x: x["catalyst"])
    df["reactant"] = df.Reaction.apply(lambda x: x["reactant"])
    df["product"] = df.Reaction.apply(lambda x: x["product"])
    df = df.drop("Reaction", axis=1)

    fn_data_original = "data_original.csv"
    df.to_csv(fn_data_original, index=False)
    del df

    # create dataframe
    df = pd.read_csv(
        fn_data_original, delimiter=","
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "Reaction_ID",
        "Y",
        "split",
        "catalyst",
        "reactant",
        "product",
    ]
    fields_clean = ["Reaction_ID", "yield", "split", "catalyst", "reactant", "product"]
    # overwrite column names = fields
    df.columns = fields_clean
    assert df.columns.tolist() == fields_clean

    # remove leading and trailing white space characters
    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "uspto_500k",
        "description": """United States Patent and Trademark Office reaction dataset with yields.""",
        "targets": [
            {
                "id": "yield",
                "description": "reaction yields",
                "units": "%",
                "type": "continuous",
                "names": [
                    "reaction yield",
                    "yield",
                ],
                "uris": [
                    "http://purl.allotrope.org/ontologies/quality#AFQ_0000227",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "reaction_SMILES",
                "type": "RXNSMILES",
                "description": "reaction SMILES",
            },
        ],
        "license": "CC0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.17863/CAM.16293",
                "description": "corresponding publication",
            },
            {
                "url": "https://github.com/reymond-group/drfp/blob/main/data/uspto_yields_below.csv",
                "description": "data source",
            },
            {
                "url": "https://github.com/reymond-group/drfp/blob/main/data/uspto_yields_above.csv",
                "description": "data source",
            },
            {
                "url": "https://tdcommons.ai/single_pred_tasks/yields/#uspto",
                "description": "other source",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "bibtex": [
            """@article{https://doi.org/10.17863/cam.16293,
doi = {10.17863/CAM.16293},
url = {https://www.repository.cam.ac.uk/handle/1810/244727},
year = {2012},
publisher = {Apollo - University of Cambridge Repository},
keywords = {Name to structure,  OPSIN,  Chemical text mining,  Text mining,
Patent reaction extraction,  Reaction mining,  Patents},
language = {en},
author = {Lowe,  Daniel Mark},
title = {Extraction of chemical structures and reactions from the literature},
copyright = {All Rights Reserved}""",
        ],
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
