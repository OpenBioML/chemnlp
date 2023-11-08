from chemnlp.data.reprs import smiles_to_iupac_name, smiles_to_safe

# not used at the moment
# def test_smiles_to_safe():
#     safe = smiles_to_safe("CC(Cc1ccc(cc1)C(C(=O)O)C)C")
#     # equivalent, only rotations, it is not completely deterministic
#     assert (
#         safe == "c12ccc3cc1.C3(C)C(=O)O.CC(C)C2"
#         or safe == "c13ccc2cc1.C2(C)C(=O)O.CC(C)C3"
#     )


def test_smiles_to_iupac_name():
    iupac_name = smiles_to_iupac_name("CC(Cc1ccc(cc1)C(C(=O)O)C)C")
    assert iupac_name == "2-[4-(2-methylpropyl)phenyl]propanoic acid"
