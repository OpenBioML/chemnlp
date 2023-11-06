from chemnlp.data.reprs import smiles_to_safe


def test_smiles_to_safe():
    safe = smiles_to_safe("CC(Cc1ccc(cc1)C(C(=O)O)C)C")
    assert safe == "c12ccc3cc1.C3(C)C(=O)O.CC(C)C2"
