import pandas as pd
import yaml
from tdc.single_pred import HTS


def get_and_transform_data():
    # get raw data
    label = "serine_threonine_kinase_33_butkiewicz"
    data = HTS(name = label)
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
        "activity_serine_threonine_kinase33",
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
    meta =  {
        "name": "serine_threonine_kinase_33_butkiewicz",  # unique identifier, we will also use this for directory names
        "description": """These are nine high-quality high-throughput screening (HTS) datasets from [1]. These datasets were curated from HTS data at the PubChem database [2]. Typically, HTS categorizes small molecules into hit, inactive, or unspecified against a certain therapeutic target. However, a compound may be falsely classified as a hit due to experimental artifacts such as optical interference. Moreover, because the screening is performed without duplicates, and the cutoff is often set loose to minimize the false negative rates, the results from the primary screens often contain high false positive rates [3]. Hence the result from the primary screen is only used as the first iteration to reduce the compound library to a smaller set of further confirmatory tests. Here each dataset is carefully collated through confirmation screens to validate active compounds. The curation process is documented in [1]. Each dataset is identified by the PubChem Assay ID (AID). Features of the datasets: (1) At least 150 confirmed active compounds present; (2) Diverse target classes; (3) Realistic (large number and highly imbalanced label).""",
        "targets": [
            {
                "id": "activity_serine_threonine_kinase33",  # name of the column in a tabular dataset
                "description": "whether it active against serine threonine kinase 33 receptor (1) or not (0).",  # description of what this column means
                "units": "activity",  # units of the values in this column (leave empty if unitless)
                "type": "categorical",  # can be "categorical", "ordinal", "continuous"
                "names": [  # names for the property (to sample from for building the prompts)
                    "serine threonine kinase 33 activity",
                    "serine threonine kinase 33 Inhibitor",
                    "activity against serine threonine kinase 33",
                    "serine threonine kinase 33 receptor"
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
