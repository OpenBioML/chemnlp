import glob
import os
import shutil

import pandas as pd
import yaml

templates = {
    "chebi_chebi": [
        """The {node1_name#} {rel1_type#} {node2_name#}.""",
    ],
    "compound_chebi": [
        """The {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} {node2_name#}.""",
        """Task: Please {#create|generate!} {#a compound |a !}{SMILES__description} that {rel1_type#} {node2_name#}.
Result: {SMILES#}""",  # noqa E501
        """Task: Please {#create|generate!} {#a compound |a !}{SMILES__description} that {rel1_type#} {node2_name#}.
Result:<EOI> {SMILES#}""",  # noqa E501
        # Add templates with constraint section?
    ],
    "compound_chebi_chebi": [
        """The {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} {node2_name#} and {rel2_type#} {node3_name#}.""",  # noqa E501
        # """Task: Please {#create|generate!} {#a compound |a !}{SMILES__description} that {rel1_type#} {node2_name#} and {rel2_type#} {node3_name#}.  # noqa E501
        # Result: {SMILES#}""",  # noqa E501
        # """Task: Please {#create|generate!} {#a compound |a !}{SMILES__description} that {rel1_type#} {node2_name#} and {rel2_type#} {node3_name#}.  # noqa E501
        # Result:<EOI> {SMILES#}""",  # noqa E501
        # Not sure if this templates make sense here? Add templates with constraint section?
    ],
    "compound_chebi_chebi_chebi": [],
    "compound_protein": [
        """The {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}.""",
        """The {node2_type#} {node2_protein_names#} is targeted by the drug with the {SMILES__description} {SMILES#}.""",  # noqa E501
        """User: Can you {#give me|come up with!} {#one|an!} example for a {node1_type#} {SMILES__description} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}.""",  # noqa E501
    ],
    "compound_protein_compound": [
        """The {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} the {node3_type#} {node3_smiles#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by the compound with the {SMILES__description} {SMILES#} and {node3_smiles#}.""",  # noqa E501
        """User: Can you {#give me|come up with!} {#one|an!} example for a {node1_type#} {SMILES__description} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}.
User: Can you {#tell|create|generate!} another {node1_type#} {SMILES__description} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: {#Sure|Yes|Of course|Yes, of course!}, the {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node3_type#} {SMILES__description} {node3_smiles#}.""",  # noqa E501
    ],
    "compound_protein_compound_3": [
        """The {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} the {node3_type#} {node3_smiles#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by the compound with the {SMILES__description} {SMILES#} and {node3_smiles#}.""",  # noqa E501
        """User: Can you {#give me|come up with!} {#one|an!} example for a {node1_type#} {SMILES__description} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}.
User: Can you {#tell|create|generate!} another {node1_type#} {SMILES__description} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: {#Sure|Yes|Of course|Yes, of course!}, the {node1_type#} {SMILES__description} {SMILES#} {rel1_type#} the {node3_type#} {SMILES__description} {node3_smiles#}.""",  # noqa E501
    ],
    "compound_protein_disease": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that is targeted by the {node1_type#} {SMILES#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node2_type#} {node2_protein_names#} is targeted by the {node1_type#} {SMILES#}.
User: Can you tell me which disease the {node2_type#} {node2_protein_names#} {rel2_type#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#} {node3_type#}.""",  # noqa E501
    ],
    "compound_protein_domain": [
        """{SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a domain of the {node2_type#} {node2_protein_names#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} a {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_ec_number": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} the {node3_name#} (EC {node3_id#}).""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. Furthermore, the {node1_type#} {SMILES#} {rel2_type#} the {node3_name#} (EC {node3_id#}).""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}.
User: Can you tell me which enzyme the {node2_type#} {node2_protein_names#} {rel2_type#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} a {node3_name#} (EC {node3_id#}).""",  # noqa E501
    ],
    # todo: There are some entries that have the EC number under node3_name and node3_id
    # and this is not handled yet properly.
    "compound_protein_go_term": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_hpo": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the human phenotype represented by {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_hpo_disease": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}. The {node3_name#} {rel3_type#} the {node4_type#} {node4_name#}.""",  # noqa E501
    ],
    "compound_protein_pathway": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "compound_protein_pathway_disease": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}. The {node3_name#} is {rel3_type#} the {node4_type#} {node4_name#}.""",  # noqa E501
    ],
    "compound_protein_protein": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_protein_names#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by {SMILES#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_protein_names#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#}?
Assistant: The {node1_type#} {SMILES#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a {node3_type#} that {rel2_type#} {node2_type#} {node2_protein_names#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node2_type#} {node2_protein_names#} {rel2_type#} {node3_protein_names#}.""",  # noqa E501
    ],
    "drug_chebi": [
        """The {node1_type#} {SMILES#} {rel1_type#} {node2_name#}.""",
    ],
    "drug_chebi_chebi": [
        """The {node1_type#} {SMILES#} {rel1_type#} {node2_name#} and {rel2_type#} {node3_name#}.""",  # noqa E501
    ],
    "drug_chebi_chebi_chebi": [],
    "drug_disease_pathway": [
        """The {node1_type#} {SMILES#|node1_name#} is indicated for the {node2_name#} {node2_type#} and {rel2_type#} the {node3_name#} {node3_type#}.""",  # noqa E501
    ],
    "drug_disease_pathway_protein": [
        """The {node1_type#} {SMILES#|node1_name#} is indicated for the {node2_name#} {node2_type#} and {rel2_type#} the {node3_name#} {node3_type#}. The {node3_type#} {node3_name#} {rel3_type#} the {node4_type#} {node4_protein_names#}.""",  # noqa E501
    ],
    "drug_protein": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#}.""",
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that is targeted by the {node1_type#} {SMILES#|node1_name#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node2_type#} {node2_protein_names#} is targeted by {#this|the above!} {node1_type#}.""",  # noqa E501
    ],
    "drug_protein_disease": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that is targeted by the {node1_type#} {SMILES#|node1_name#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node2_type#} {node2_protein_names#} is targeted by {#this|the above!} {node1_type#}.
User: Can you tell me which disease the {node2_type#} {node2_protein_names#} {rel2_type#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#} {node3_type#}.""",  # noqa E501
    ],
    "drug_protein_domain": [
        """{SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} a {node3_name#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#|node1_name#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a domain of the {node2_type#} {node2_protein_names#}?
Assistant: The {node2_type#} {node2_protein_names#} {rel2_type#} a {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_drug": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_type#} {node3_name#|node3_smiles#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by the drugs {SMILES#|node1_name#} and {node3_name#|node3_smiles#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a {node1_type#} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: {#Sure|Yes|Of course|Yes, of course!}, the {node1_type#} {rel1_type#} the {node2_type#} {node2_protein_names#}.
User: Can you tell me another {node1_type#} that {rel1_type#} the {node2_type#} {node2_protein_names#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node3_type#} {node3_name#|node3_smiles#}.""",  # noqa E501
    ],
    "drug_protein_ec_number": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_name#} which {rel2_type#} the {node3_name#} (EC {node3_id#}) reaction.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_name#}. Furthermore, the {node2_type#} {node2_name#} {rel2_type#} the {node3_name#} (EC {node3_id#}) reaction.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#|node1_name#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_name#}.
User: Can you tell me which reaction the {node2_type#} {node2_name#} {rel2_type#}?
Assistant: The {node2_type#} {node2_name#} {rel2_type#} a {node3_name#} (EC {node3_id#}) reaction.""",  # noqa E501
    ],
    "drug_protein_go_term": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_name#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_name#}. The {node2_type#} {node2_name#} {rel2_type#} the {node3_name#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#|node1_name#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_name#}.
User: Can you tell me more {#details |!}about {node2_type#} {node2_name#}?
Assistant: {#Sure|Yes|Of course|Yes, of course!}, the {node2_type#} {node2_name#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_hpo": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#} and {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the human phenotype represented by {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#}. This {node2_type#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_hpo_disease": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}. The {node3_name#} {rel3_type#} the {node4_type#} {node4_name#}.""",  # noqa E501
    ],
    "drug_protein_pathway": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_name#}.""",  # noqa E501
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#|node1_name#}?
Assistant: {#Sure|Yes|Of course|Yes, of course!}, the {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_name#}.
User: Can you tell me more {#details |!}about {node2_type#} {node2_name#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node2_type#} {node2_name#} {rel2_type#} the {node3_name#}.""",  # noqa E501
    ],
    "drug_protein_pathway_disease": [
        """The {node1_type#} {SMILES#} {rel1_type#} the {node2_type#} {node2_protein_names#}. The {node2_type#} {node2_protein_names#} {rel2_type#} the {node3_name#}. The {node3_name#} is {rel3_type#} the {node4_type#} {node4_name#}.""",  # noqa E501
    ],
    "drug_protein_protein": [
        """The {node1_type#} {SMILES#|node1_name#} {rel1_type#} the {node2_type#} {node2_protein_names#} which {rel2_type#} the {node3_type#} {node3_name#}.""",  # noqa E501
        """The {node2_type#} {node2_protein_names#} is targeted by {SMILES#|node1_name#}. The {node2_type#} {node2_protein_names#} {rel2_type#} {node3_name#}.""",  # noqa E501
        """User: {#Can you give me|Can you come up with!} {#an|one!} example for a protein that binds the {node1_type#} {SMILES#|node1_name#}?
Assistant: The {node1_type#} {SMILES#|node1_name#} {rel1_type#} for example the {node2_type#} {node2_protein_names#}.
User: Can you tell me a {node3_type#} that {rel2_type#} {node2_type#} {node2_protein_names#}?
Assistant: {#Yes|Of course|Yes, of course|Sure!}, the {node2_type#} {node2_name#} {rel2_type#} {node3_type#} {node3_name#}.""",  # noqa E501
    ],
    "chembl33_preprocessed_filtered_bioactivity_dataset_w_fullprotnames_smiles": [
        "The {#molecule with the |!}{SMILES__description} {#representation of |!}{SMILES#} {#shows|exhibits|displays!} a {#bioaffinity|affinity!} for {#the protein |!}{protein_name#} with a {standard_type#} {#value |!}of {standard_value#} {standard_units#}.",  # noqa E501
        # Instruction tuning text templates
        """Task: Please {#derive|estimate|approximate!} {#the bioaffinity|the affinity!} of a {#molecule to a protein|protein to a molecule!}.
Protein{# name|!}: {protein_name#}
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint{#s|!}: The {#resulting|derived|calculated!} {standard_type#} {#value |!}should be in {standard_units#}. Even if you are {#uncertain|not sure!}, you must {#derive|estimate|come up with!} a {standard_type#} {#value |!}without using any {#other|additional!} words.
Result: {standard_value#} {standard_units#}""",  # noqa E501
        """Task: Please {#create|generate!} {#a molecule |a !}{SMILES__description} that has a {#bioaffinity|affinity!} to {#the protein |!}{protein_name#} with a {standard_type#} {#value |!}of {standard_value#} {standard_units#}.
Result: {SMILES#}""",  # noqa E501
        # Conversational text templates
        """User: Can you {#give me|come up with!} {#one|an!} example of a protein that has a {#bioaffinity|affinity!} to the {SMILES__description} {SMILES#}?
Assistant: {#For example, the protein |For example, |!}{protein_name#} has a {#bioaffinity|affinity!} to the {SMILES__description} {SMILES#}.
User: Can you {#derive|estimate|approximate!} the {standard_type#} {#of this molecule|of this molecule for me|for me!}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, the {standard_type#} {#value |!}is {standard_value#} {standard_units#}.""",  # noqa E501
        """User: Can you {#give me|come up with!} {#one|an!} example of a protein that has a {#bioaffinity|affinity!} to the {SMILES__description} {SMILES#}?
Assistant: {#The protein |!}{protein_name#} has for example a {#bioaffinity|affinity!} to the {SMILES__description} {SMILES#}.
User: Can you {#derive|estimate|approximate!} the {standard_type#} {#of this molecule|of this molecule for me|for me!}?
Assistant: {#Yes|Of course|Sure|Yes, I'm happy to help!}, the {standard_type#} {#value |!}is {standard_value#} {standard_units#}.
User: Can you give {#me |!}{#additional|more!} {#information|details!} {#on|about!} the assay{# used| used for this estimation!}?
Assistant: {#Sure|Yes|Of course!}, here you go:
{description#}""",  # noqa E501
        # Benchmarking text templates, take super long to process, needs investigation
        """Task: Please {#derive|estimate|approximate!} {#the bioaffinity|the affinity!} of a {#molecule to a protein|protein to a molecule!}.
Protein{# name|!}: {protein_name#}
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint{#s|!}: The {#resulting|derived|calculated!} {standard_type#} {#value |!}should be in {standard_units#}. Even if you are {#uncertain|not sure!}, you must {#derive|estimate|come up with!} a {standard_type#} {#value |!}without using any {#other|additional!} words.
Result:<EOI> {standard_value#} {standard_units#}""",  # noqa E501
        """Task: Please {#create|generate!} a {#molecule |!}{SMILES__description} that has a {#bioaffinity|affinity!} to {#the protein |!}{protein_name#} with a {standard_type#} {#value |!}of {standard_value#} {standard_units#}.
Result:<EOI> {SMILES#}""",  # noqa E501
        """Task: Please answer the multiple choice question.
Question: What is the {#the bioaffinity|the affinity!} of a {#molecule to a protein|protein to a molecule!}?
Protein{# name|!}: {protein_name#}
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: The {#shown|listed!} {standard_type#} values {#below |!}are in {standard_units#}. Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%3-5%aA1} without using any other words.
Options:
{standard_value%}
Answer: {%multiple_choice_result}""",  # noqa E501
        """Task: Please answer the multiple choice question.
Question: What is the {#the bioaffinity|the affinity!} of a {#molecule to a protein|protein to a molecule!}?
Protein{# name|!}: {protein_name#}
{#Molecule |!}{SMILES__description}: {SMILES#}
Constraint: The {#shown|listed!} {standard_type#} values {#below |!}are in {standard_units#}. Even if you are {#uncertain|not sure!}, you must pick either {%multiple_choice_enum%3-5%aA1} without using any other words.
Options:
{standard_value%}
Answer:<EOI> {%multiple_choice_result}""",  # noqa E501
    ],
}

# create meta yaml
meta_template = {
    "name": None,
    "description": "Knowledgegraph data samples.",
    "targets": None,
    "identifiers": None,
    "license": "CC BY 4.0",
    "links": [
        {
            "url": "https://crossbar.kansil.org",
            "description": "original knowledge graph web GUI link",
        },
    ],
    "num_points": None,
    "bibtex": [
        """@article{10.1093/nar/gkab543,
author = {Doğan, Tunca and Atas, Heval and Joshi, Vishal and Atakan, Ahmet and Rifaioglu, Ahmet Sureyya and Nalbat, Esra and Nightingale, Andrew and Saidi, Rabie and Volynkin, Vladimir and Zellner, Hermann and Cetin-Atalay, Rengul and Martin, Maria and Atalay, Volkan},
title = "{CROssBAR: comprehensive resource of biomedical relations with knowledge graph representations}",
journal = {Nucleic Acids Research},
volume = {49},
number = {16},
pages = {e96-e96},
year = {2021},
month = {06},
issn = {0305-1048},
doi = {10.1093/nar/gkab543},
url = {https://doi.org/10.1093/nar/gkab543},
}"""  # noqa E501
    ],
    "templates": None,
}


recode = {
    "Drug": "drug",
    "Protein": "protein",
    "Pathway": "pathway",
    "Compound": "compound",
    "Disease": "disease",
    "Abnormality": "abnormality",
    "has_functional_parent": "has the functional parent",
    "has_parent_hydride": "has the parent hydride",
    "has_part": "has the part",
    "has_role": "has the role",
    "interacts_with": "interacts with",
    "involved_in": "is involved in",
    "is_a": "is a",
    "is_associated_with": "is associated with",
    "is_conjugate_acid_of": "is the conjugate acid of",
    "is_conjugate_base_of": "is the conjugate based of",
    "is_enantiomer_of": "is a enantiomer of",
    "is_involved_in": "is involved in",
    "is_ortholog_to": "is ortholog to",
    "is_related_to": "is related to",
    "is_substituent_group_from": "is the substituent group from",
    "is_tautomer_of": "is a tautomer of",
    "located_in": "is located in",
    "is_targeted_by": "which is also targeted by",
    "modulated_by": "modulated by",
    "Drug metabolism": "drug metabolism",
    "Calcium": "calcium",
    "Rectal fistula": "rectal fistula",
}


def create_yamls(dirs: list):
    """Creates meta.yamls based on the data_clean.csv files
    and the text templates from the template dict."""
    for path in dirs:
        # subselect one path
        # if path.find("drug_protein_pathway_disease") == -1: continue
        df = pd.read_csv(
            path + "data_clean.csv",
            index_col=False,
        )
        cols = df.columns.tolist()

        dataset_name = path.split("/")[-2]
        meta_copy = meta_template.copy()
        meta_copy["name"] = dataset_name
        meta_copy["num_points"] = len(df)
        if "SMILES" in cols:
            meta_copy["identifiers"] = [
                {"id": "SMILES", "description": "SMILES", "type": "SMILES"}
            ]
            meta_copy["targets"] = [
                {
                    "id": c,
                    "description": c,
                    "type": "Other",
                    "units": c,
                    "names": [{"noun": c}],
                }
                for c in cols
                if c != "SMILES"
            ]
        else:
            # for KG walks data w/o SMILES column
            meta_copy["identifiers"] = [
                {"id": c, "description": c, "type": "Other"} for c in cols if "1" in c
            ]
            meta_copy["targets"] = [
                {
                    "id": c,
                    "description": c,
                    "type": "Other",
                    "units": c,
                    "names": [{"noun": c}],
                }
                for c in cols
                if "1" not in c
            ]

        if dataset_name in templates:
            meta_copy["templates"] = templates[dataset_name]
        else:
            # get dataset name from split up datasets w/o count integer at the end
            dataset_name_short = [
                dn
                for dn in templates
                if dn.startswith("_".join(dataset_name.split("_")[:-1]))
            ][0]
            meta_copy["templates"] = templates[dataset_name_short]

        fn_meta = path + "meta.yaml"
        with open(fn_meta, "w") as f:
            yaml.dump(meta_copy, f, sort_keys=False)
        print(dataset_name)


def format_kg_df(df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Formats a pandas dataframe."""
    df.drop_duplicates(inplace=True)

    # replace the empty string with NaN
    df.replace("", pd.NA, inplace=True)

    # replace CHEMBL... strings with NaN
    # df = df.applymap(lambda x: pd.NA if str(x).startswith("CHEMBL") else x)

    # recode based on lookup dict
    df.replace(recode, inplace=True)

    # check if KG assay export
    if "Assay_CHEMBL_ID" in df.columns:
        # remove entries with None values in specific columns used in the templates
        df.dropna(subset=["Protein Name"], inplace=True)
        df.dropna(subset=["standard_type"], inplace=True)
        df.dropna(subset=["standard_value"], inplace=True)
        df.dropna(subset=["standard_units"], inplace=True)
        df.dropna(subset=["description"], inplace=True)

        # drop columns we don't use in the templates
        df.drop(
            columns=[
                "Target_CHEMBL_ID",
                "Compound_CHEMBL_ID",
                "Assay_CHEMBL_ID",
                "assay_type",
                "Assay Taxonomy",
                "TD Tax ID",
                "confidence_score",
                "target_type",
                "src_compound_id",
                "src_assay_id",
                "src_id",
                "src_description",
                "standard_relation",
                "activity_comment",
                "year",
            ],
            inplace=True,
        )

        df.columns = [
            c.lower().replace(" ", "_") if c != "SMILES" else "SMILES"
            for c in df.columns
        ]
        return df
    elif "node1_smiles" in df.columns:
        df.columns = [c if c != "node1_smiles" else "SMILES" for c in df.columns]

    # relations to lower caser
    if "node2_type" in df.columns:
        df.node2_type = df.node2_type.apply(lambda x: x.lower())

    if "node3_type" in df.columns:
        df.node3_type = df.node3_type.apply(lambda x: x.lower())

    # for drug_protein_drug
    if "rel2_type" in df.columns:
        df.rel2_type.replace({"targets": "which is also targeted by"}, inplace=True)

    if "node3_type" in df.columns:
        if df.node3_type.unique().tolist() == ["Domain"]:
            df.node3_name = df.node3_name.apply(lambda x: x.replace(",", ""))

            def check_and_add_domain(x):
                if "domain" in x.lower():
                    return x
                else:
                    return x + " domain"

            df.node3_name = df.node3_name.apply(check_and_add_domain)

    # for drug_chebi_chebi
    if "node3_type" in df.columns:
        if (df.node2_type.unique().tolist() == ["chebi"]) and (
            df.node3_type.unique().tolist() == ["chebi"]
        ):
            df.drop(
                labels=df[
                    df.node1_name.apply(lambda x: x.lower())
                    == df.node2_name.apply(lambda x: x.lower())
                ].index.tolist(),
                inplace=True,
            )

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    # for compound_protein_compound
    # if "node2_name" in df.columns:
    #    df["node2_name"] = df["node2_name"].apply(
    #        lambda x: pd.NA if str(x).startswith("CHEMBL") else x
    #    )
    # if "node3_name" in df.columns:
    #    df["node3_name"] = df["node3_name"].apply(
    #        lambda x: pd.NA if str(x).startswith("CHEMBL") else x
    #    )

    # for compound_protein_disease
    if "node3_name" in df.columns:

        def not_nan_string_check(x, s):
            try:
                if x == s:
                    return True
                else:
                    return False
            except Exception:
                return False

        df["node3_name"] = df["node3_name"].apply(
            lambda x: pd.NA if not_nan_string_check(x, "GM00719") else x
        )

    # for compound_protein_ec_number
    if "node3_name" in df.columns:

        def not_nan_digit_check(x):
            """Function returns true for ec number, e.g., 4.2.1.-,
            by identifying it by checking if the first character is a digit."""
            try:
                if x[0].isdigit():
                    return True
                else:
                    return False
            except Exception:
                return False

        df["node3_name"] = df["node3_name"].apply(
            lambda x: pd.NA if not_nan_digit_check(x) else x
        )

    # remove NaN rows
    print(f"w/  NaNs: {len(df)}")
    df.dropna(inplace=True)
    print(f"w/o NaNs: {len(df)}")

    return df


def str_presenter(dumper, data: str):
    """configures yaml for dumping multiline strings
    Ref: https://stackoverflow.com/questions/8640959/how-can-i-control-what-scalar-form-pyyaml-uses-for-my-data
    """
    if data.count("\n") > 0:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def preprocess_kg_data(path_data_dir: str):
    """Preprocesses the raw knowledge graph data and save
    the data_original.csv, data_clean.csv, and meta.yaml."""
    # create separate dirs, move csv files there, and save cleaned data
    fns_data_raw = sorted(glob.glob(path_data_dir + "*.csv"))  # KG walks data
    fns_data_raw += sorted(glob.glob(path_data_dir + "*.tsv"))  # KG assay data

    for fn in fns_data_raw:
        # subselect one path
        # if fn.find("Drug_Protein_Pathway_Disease") == -1: continue
        if fn.endswith("_mappings.csv") or fn.endswith("_full.csv"):
            continue
        print(f"\n###### {fn}")
        dir_new = (
            fn.split("/")[-1]
            .split(".csv" if fn.endswith(".csv") else ".tsv")[0]
            .lower()
        )
        path_new = path_data_dir + dir_new
        os.makedirs(path_new, exist_ok=True)
        path_data_original = (
            path_new + "/data_original" + (".csv" if fn.endswith(".csv") else ".tsv")
        )
        shutil.copyfile(fn, path_data_original)
        df = pd.read_csv(
            path_data_original,
            low_memory=False,
            index_col=False,
            sep="," if fn.endswith(".csv") else "\t",
        )
        df = format_kg_df(df)
        path_data_clean = (
            path_data_original
            if fn.endswith(".csv")
            else path_data_original.replace(".tsv", ".csv")
        ).replace("_original.csv", "_clean.csv")
        if len(df) > 0:
            df.to_csv(path_data_clean, index=False)
        else:
            shutil.rmtree(path_new)
            print(f"Removed {path_new}!")

    # set up yamls
    dirs = sorted(glob.glob(path_data_dir + "*/", recursive=False))

    yaml.add_representer(str, str_presenter)
    yaml.representer.SafeRepresenter.add_representer(str, str_presenter)

    create_yamls(dirs)


if __name__ == "__main__":
    path_data_dir = __file__.replace("text_sampling/preprocess_kg.py", "kg/")
    preprocess_kg_data(path_data_dir)
