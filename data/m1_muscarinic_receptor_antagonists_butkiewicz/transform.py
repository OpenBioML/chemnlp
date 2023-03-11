import pandas as pd
import yaml
from tdc.single_pred import HTS


def get_and_transform_data():
    # get raw data
    label = "m1_muscarinic_receptor_antagonists_butkiewicz"
    data = HTS(name=label)
    fn_data_original = "data_original.csv"
    data.get_data().to_csv(fn_data_original, index=False)

    # create dataframe
    df = pd.read_csv(
        fn_data_original,
        delimiter=",",
    )  # not necessary but ensure we can load the saved data

    # check if fields are the same
    fields_orig = df.columns.tolist()
    assert fields_orig == [
        "Drug_ID",
        "Drug",
        "Y",
    ]

    # overwrite column names = fields
    fields_clean = [
        "compound_id",
        "SMILES",
        "m1_muscarinic_antagonists",
    ]
    df.columns = fields_clean

    #     # data cleaning
    #     df.compound_id = (
    #         df.compound_id.str.strip()
    #     )  # remove leading and trailing white space characters

    assert not df.duplicated().sum()

    # save to csv
    fn_data_csv = "data_clean.csv"
    df.to_csv(fn_data_csv, index=False)

    # create meta yaml
    meta = {
        "name": "m1_muscarinic_receptor_antagonists_butkiewicz",  # unique identifier, we will also use this for directory names
        "description": """""",
        "targets": [
            {
                "id": "m1_muscarinic_antagonists",  # name of the column in a tabular dataset
                "description": "whether it antagonists on m1 muscarinic receptor (1) or not (0).",  # description of what this column means
                "units": "antagonists",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "m1 muscarinic activity",
                    "m1 muscarinic antagonists",
                    "muscarinic",
                    "m1 muscarinic receptor",
                ],
            },
        ],
        "identifiers": [
            {
                "id": "SMILES",  # column name
                "type": "SMILES",  # can be "SMILES", "SELFIES", "IUPAC", "Other"
                "description": "SMILES",  # description (optional, except for "Other")
            },
        ],
        "license": "CC BY 4.0",  # license under which the original dataset was published
        "links": [  # list of relevant links (original dataset, other uses, etc.)
            {
                "url": "https://doi.org/10.3390/molecules18010735",
                "description": "corresponding publication",
            },
            {
                "url": "https://doi.org/10.1093/nar/gky1033",
                "description": "corresponding publication",
            },
            {
                "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5962024/",
                "description": "corresponding publication",
            },
        ],
        "num_points": len(df),  # number of datapoints in this dataset
        "url": "https://tdcommons.ai/single_pred_tasks/hts/#butkiewicz-et-al",
        "bibtex": [
            """@article{Butkiewicz2013,
              doi = {10.3390/molecules18010735},
              url = {https://doi.org/10.3390/molecules18010735},
              year = {2013},
              month = jan,
              publisher = {{MDPI} {AG}},
              volume = {18},
              number = {1},
              pages = {735--756},
              author = {Mariusz Butkiewicz and Edward Lowe and Ralf Mueller and Jeffrey Mendenhall and Pedro Teixeira and C. Weaver and Jens Meiler},
              title = {Benchmarking Ligand-Based Virtual High-Throughput Screening with the {PubChem} Database},
              journal = {Molecules}}""",
            """@article{Kim2018,
              doi = {10.1093/nar/gky1033},
              url = {https://doi.org/10.1093/nar/gky1033},
              year = {2018},
              month = oct,
              publisher = {Oxford University Press ({OUP})},
              volume = {47},
              number = {D1},
              pages = {D1102--D1109},
              author = {Sunghwan Kim and Jie Chen and Tiejun Cheng and Asta Gindulyte and Jia He and Siqian He and Qingliang Li and Benjamin A Shoemaker and Paul A Thiessen and Bo Yu and Leonid Zaslavsky and Jian Zhang and Evan E Bolton},
              title = {{PubChem} 2019 update: improved access to chemical data},
              journal = {Nucleic Acids Research}}""",
            """@article{Butkiewicz2017,
              doi = {},
              url = {https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5962024/},
              year = {2017},
              publisher = {Chem Inform},
              volume = {3},
              number = {1},
              author = {Butkiewicz, M.  and Wang, Y.  and Bryant, S. H.  and Lowe, E. W.  and Weaver, D. C.  and Meiler, J.},
              title = {{H}igh-{T}hroughput {S}creening {A}ssay {D}atasets from the {P}ub{C}hem {D}atabase}},
              journal = {Chemical Science}}""",
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
