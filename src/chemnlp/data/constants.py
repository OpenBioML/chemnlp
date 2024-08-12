DEFAULT_SIGNIFICANT_DIGITS = 3


STANDARD_TABULAR_TEXT_TEMPLATES = [
    "The molecule with the {SMILES__description} {#representation of |!}{SMILES#} has a {TARGET__names__noun} of {TARGET#} {TARGET__units}.",  # noqa: E501
    "Based on the {SMILES__description} {#representation of |!}{SMILES#}, the molecule has a {TARGET__names__noun} of {TARGET#} {TARGET__units}.",  # noqa: E501
    "The {SMILES__description} {SMILES#} {#represents|is representing!} a molecule {#that has a|with a!} {TARGET__names__noun} of {TARGET#} {TARGET__units}.",  # noqa: E501
    "The molecule with the {SMILES__description} {SMILES#} has a {TARGET__names__noun} of {TARGET#} {TARGET__units}.",
    # Instruction tuning text templates
    """Task: Please predict a molecule feature based on the description.
Description: Predict the {TARGET__names__noun} in {TARGET__units}.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {TARGET__units} without using any {#other|additional!} words.
Result: {TARGET#} {TARGET__units}""",  # noqa: E501
    """Task: Please predict a molecule feature based on the description.
Description: Predict the {TARGET__names__noun} in {TARGET__units}.
{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {TARGET__units} without the unit and without using any {#other|additional!} words.
Result: {TARGET#}""",  # noqa: E501
    """Task: Please {#give me|create|generate!} a {#molecule|chemical|compound!} with {SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has a {TARGET__names__noun} of {TARGET#} {TARGET__units}.
Result: {SMILES#}""",  # noqa: E501
    # Conversational text templates
    """User: Can you {#tell me|derive|estimate!} the {TARGET__names__noun} in {TARGET__units} of the molecule with the {SMILES__description} {SMILES#}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, this molecule has a {TARGET__names__noun} of {TARGET#} {TARGET__units}.""",  # noqa: E501
    """User: Can you {#give me|create|generate!} the {SMILES__description} of a molecule that has a {TARGET__names__noun} of {TARGET#} {TARGET__units}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, here you go: {SMILES#}""",  # noqa: E501
    """User: I'm {#searching|looking!} for the {SMILES__description} of a molecule that has a {TARGET__names__noun} of {TARGET#} {TARGET__units}.
Assistant: This is a molecule that has a {TARGET__names__noun} of {TARGET#} {TARGET__units}: {SMILES#}""",  # noqa: E501
    """User: I want to {#come up with|create|generate!} the {SMILES__description} of a {#molecule|chemical|chemical compound!}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should I consider any {#constraints|specific points!} for the {#generation|creation!}?
User: Yes, please. The molecule should have a {TARGET__names__noun} of {TARGET#} {TARGET__units}.
Assistant: {#Ok|Got it!},{# here you go,|!} this {SMILES__description} represents a molecule that has a {TARGET__names__noun} of {TARGET#} {TARGET__units}: {SMILES#}""",  # noqa: E501
    """User: I want to {#come up with|create|generate!} a {SMILES__description} of a {#molecule|chemical|chemical structure!}.
Assistant: {#This sounds very exciting. |This sounds very interesting. !}Should it be a special {#molecule|one!}?
User: Yes, the molecule should have a {TARGET__names__noun} of {TARGET#} {TARGET__units}.
Assistant: {#Understood|Got it|Ok!}, this {SMILES__description} represents a molecule that has a {TARGET__names__noun} of {TARGET#} {TARGET__units}: {SMILES#}""",  # noqa: E501
    # Benchmarking text templates
    "The {TARGET__names__noun} of the molecule with the {SMILES__description} {SMILES#} is:<EOI>{TARGET#} {TARGET__units}",  # noqa: E501
    "The {TARGET__names__noun} of the {SMILES__description} {SMILES#} is:<EOI>{TARGET#} {TARGET__units}",  # noqa: E501
    "The {TARGET__names__noun} of the molecule {SMILES__description} {SMILES#} is:<EOI>{TARGET#} {TARGET__units}",  # noqa: E501
    """Task: Please predict a molecule feature based on the description.
Description: Predict the {TARGET__names__noun} in {TARGET__units} of a molecule.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {TARGET__units} without using any {#other|additional!} words.
Result:<EOI>{TARGET#} {TARGET__units}""",  # noqa: E501
    """Task: Please predict a molecule feature based on the description.
Description: Predict the {TARGET__names__noun} in {TARGET__units} of a molecule.
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: Even if you are {#uncertain|not sure!}, you must answer with a numeric value in {TARGET__units} without the unit and without using any {#other|additional!} words.
Result:<EOI>{TARGET#}""",  # noqa: E501
    """Task: Please {#give me|create|generate!} a {#molecule |!}{SMILES__description} based on the {#text |!}description{# below|!}.
Description: A molecule that has a {TARGET__names__noun} of {TARGET#} {TARGET__units}.
Result:<EOI>{SMILES#}""",  # noqa: E501
]


EXCLUDE_FROM_STANDARD_TABULAR_TEXT_TEMPLATES = [
    "BACE",
    "BBBP",  # because it is boolean target data
    "MUV_466",  # boolean target data
    "MUV_548",  # boolean target data
    "MUV_600",  # boolean target data
    "MUV_644",  # boolean target data
    "MUV_652",  # boolean target data
    "MUV_689",  # boolean target data
    "MUV_692",  # boolean target data
    "MUV_712",  # boolean target data
    "MUV_713",  # boolean target data
    "MUV_733",  # boolean target data
    "MUV_737",  # boolean target data
    "MUV_810",  # boolean target data
    "MUV_832",  # boolean target data
    "MUV_846",  # boolean target data
    "MUV_852",  # boolean target data
    "MUV_858",  # boolean target data
    "MUV_859",  # boolean target data
    "RedDB",
    "SIDER",
    "ames_mutagenicity",  # because it is boolean target data
    "aminoacids",
    "bc5chem",
    "bc5disease",
    "bicerano_dataset",
    "bio_ner",
    "bioavailability_ma_et_al",  # because it is boolean target data
    "block_polymers_morphology",
    "blood_brain_barrier_martins_et_al",  # because it is boolean target data
    "buchwald_hartwig",
    "carcinogens",  # because it is boolean target data
    "cav3_t-type_calcium_channels_butkiewicz",  # because it is boolean target data
    "chebi_20",  # target is text description
    "chem_caption_smarts",
    "chembl_v29",  # text only, no SMILES
    "chemcaption_fragments",
    "chemcaption_rdkit",  # text only, no SMILES
    "chemdner",
    "chemistry_stackexchange",
    "choline_transporter_butkiewicz",  # because it is boolean target data
    "clintox",  # because it is boolean target data
    "compound_chebi_chebi_chebi_1",
    "compound_chebi_chebi_chebi_2",
    "core_mof_no_topo",
    "cyp2c9_substrate_carbonmangels",  # boolean target data
    "cyp2d6_substrate_carbonmangels",  # boolean target data
    "cyp3a4_substrate_carbonmangels",  # boolean target data
    "cyp_p450_1a2_inhibition_veith_et_al",  # boolean target data
    "cyp_p450_2c19_inhibition_veith_et_al",  # boolean target data
    "cyp_p450_2c9_inhibition_veith_et_al",  # boolean target data
    "cyp_p450_2d6_inhibition_veith_et_al",  # boolean target data
    "cyp_p450_3a4_inhibition_veith_et_al",  # boolean target data
    "drug_chebi_chebi_chebi",
    "drug_induced_liver_injury",  # boolean target data
    "drugchat_liang_zhang_et_al",  # text
    "fda_adverse_reactions",
    "formation_energies",
    "freesolv",  # more than one target
    "h2_storage_materials",
    "herg_blockers",  # more than one target
    "herg_central_inhib",  # boolean target data
    "herg_karim_et_al",  # boolean target data
    "hiv",  # boolean target data
    "human_intestinal_absorption",  # boolean target data
    "iupac_goldbook",  # text only, no SMILES
    "iupac_smiles",  # translation from IUPAC name to SMILES
    "kcnq2_potassium_channel_butkiewicz",  # boolean target data
    "m1_muscarinic_receptor_agonists_butkiewicz",  # boolean target data
    "m1_muscarinic_receptor_antagonists_butkiewicz",  # boolean target data
    "mattermodeling_stackexchange",
    "melting_points",
    "mofdscribe",
    "mol2svg",
    "mol_repr_transl_canonical_inchi",
    "mol_repr_transl_canonical_iupac_name",
    "mol_repr_transl_deepsmiles_canonical",
    "mol_repr_transl_deepsmiles_inchi",
    "mol_repr_transl_deepsmiles_iupac_name",
    "mol_repr_transl_inchi_iupac_name",
    "mol_repr_transl_selfies_canonical",
    "mol_repr_transl_selfies_deepsmiles",
    "mol_repr_transl_selfies_inchi",
    "mol_repr_transl_selfies_iupac_name",
    "mol_repr_transl_smiles_canonical",
    "mol_repr_transl_smiles_deepsmiles",
    "mol_repr_transl_smiles_inchi",
    "mol_repr_transl_smiles_iupac_name",
    "mol_repr_transl_smiles_selfies",
    "mona",  # more than one target
    "moses",
    "moses",  # SMILES only, has no target
    "mp_anisotropy",
    "mp_bulk_modulus",
    "mp_descriptions",
    "mp_self_supervised",
    "mp_shear_modulus",
    "ncbi_disease",
    "nlmchem",  # text only, no SMILES
    "nomad_structure",
    "nr_ahr_tox21",  # boolean target data
    "nr_ar_lbd_tox21",  # boolean target data
    "nr_ar_tox21",  # boolean target data
    "nr_aromatase_tox21",  # boolean target data
    "nr_er_lbd_tox21",  # boolean target data
    "nr_er_tox21",  # boolean target data
    "nr_ppar_gamma_tox21",  # boolean target data
    "ocp",
    "odd_one_out",
    "opv",
    "oqmd",
    "orbnet_denali",  # only makes sense for the structure files
    "ord_masked",
    "ord_predictions",
    "ord_procedure_steps",
    "ord_rxn_smiles_procedure",
    "ord_rxn_smiles_yield_pred",
    "ord_steps_yield",
    "orexin1_receptor_butkiewicz",  # boolean target data
    "p_glycoprotein_inhibition_broccatelli_et_al",  # boolean target data
    "pampa_ncats",  # boolean target data
    "peptides_hemolytic",  # boolean target data
    "peptides_nonfouling",  # boolean target data
    "peptides_soluble",  # boolean target data
    "perovskite_db",
    "physics_stackexchange",
    "potassium_ion_channel_kir2_1_butkiewicz",  # boolean target data
    "qm8",
    "qm9",
    "qmof_gcmc",
    "qmof_quantum",
    "rhea_db_masked",
    "rhea_db_predictions",
    "sarscov2_3clpro_diamond",  # boolean target data
    "sarscov2_vitro_touret",  # boolean target data
    "serine_threonine_kinase_33_butkiewicz",  # boolean target data
    "skin_reaction",  # boolean target data
    "smiles_to_3d",
    "sr_are_tox21",  # boolean target data
    "sr_atad5_tox21",  # boolean target data
    "sr_hse_tox21",  # boolean target data
    "sr_mmp_tox21",  # boolean target data
    "sr_p53_tox21",  # boolean target data
    "suzuki_miyaura_sach",
    "tyrosyl-dna_phosphodiesterase_butkiewicz",  # boolean target data
    "uniprot_binding_single",
    "uniprot_binding_sites_multiple",
    "uniprot_organisms",
    "uniprot_reactions",
    "uniprot_sentences",
    "uspto",
    "uspto_yield",
    "zinc",  # SMILES only, has no target
    # "h2_storage_materials",  # only IUPAC identifier, more than one target, LOW PRIO: has only 30 samples
]


LM_EVAL_YAML_TEAMPLTE_LOGLIKELIHOOD = {
    "group": [
        "chemnlp",
        "loglikelihood",
    ],
    "task": None,
    "dataset_path": None,
    "dataset_name": None,
    "output_type": "loglikelihood",
    "doc_to_text": "input",
    "doc_to_target": "output",
    "metric_list": [
        {
            "metric": "perplexity",
            "aggregation": "perplexity",
            "higher_is_better": False,
        },
        {
            "metric": "acc",
            "aggregation": "mean",
            "higher_is_better": True,
        },
    ],
}

LM_EVAL_YAML_TEMPLATE_MULTIPLE_CHOICE = {
    "group": [
        "chemnlp",
        "multiple_choice",
    ],
    "task": None,
    "dataset_path": None,
    "dataset_name": None,
    "output_type": "multiple_choice",
    "doc_to_text": "input",
    "doc_to_target": "output",
    "doc_to_choice": "{{answer_choices}}",
    "metric_list": [
        {
            "metric": "acc",
            "aggregation": "mean",
            "higher_is_better": True,
        },
        {
            "metric": "acc_norm",
            "aggregation": "mean",
            "higher_is_better": True,
        },
        # todo: check acc_mutual_info because it breaks
        # {
        #     "metric": "acc_mutual_info",
        #     "aggregation": "mean",
        #     "higher_is_better": True,
        # },
    ],
}
