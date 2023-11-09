import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tdc.generation import MolGen
from tqdm import tqdm

"""
Creating the odd-one-out dataset, for an example application such as below:

"Q: Which molecule is the least similar to the others?
1. [Mol 1]
2. [Mol 2]
3. [Mol 3]
4. [Mol 4]

A: [Answer]"

Uses Tanimoto Similarity with Morgan Fingerprints implemented in RDKit, with radius 2.
"""


def load_dataset():
    return MolGen(name="ChEMBL_V29").get_split()


def transform_dataset(dataset, n_permutations):
    smis = dataset["smiles"].values
    # Make sure smi_idx_arr contain no duplicate indexes in any of its rows
    smi_idx_arr = np.stack([np.arange(len(dataset)) for _ in range(n_permutations)]).T
    for i in range(n_permutations):
        smi_idx_arr[:, i] += i
        smi_idx_arr[:, i] %= len(dataset)

    odd_one_out_idx = []  # the least similar index, or the "odd-one-out"

    for row in tqdm(smi_idx_arr):
        try:
            # gather Morgan fingerprints for all mols in row, with radius of 2
            fingerprints = []
            for val in row:
                mol = Chem.MolFromSmiles(smis[val])
                fingerprint = AllChem.GetMorganFingerprint(mol, 2)
                fingerprints.append(fingerprint)

            # Calculate summed Tanimoto similarity of mol i to all remaining mols
            similarities = []
            for i in range(len(fingerprints)):
                similarity_sum = 0
                for j in range(len(fingerprints)):
                    if i != j:
                        similarity_sum += DataStructs.TanimotoSimilarity(
                            fingerprints[i], fingerprints[j]
                        )
                similarities.append(similarity_sum)

            lowest_similarity = np.argmin(similarities)
            odd_one_out_idx.append(lowest_similarity)
        except Exception:
            odd_one_out_idx.append(np.nan)
    return smi_idx_arr, odd_one_out_idx


def save_dataset_title():
    with open("data_clean.csv", "w") as f:
        col_strs = [f"mol_{i}" for i in range(n_permutations)]
        f.write(
            ",".join(col_strs) + ",least_similar_index,split\n"
        )  # 'mol_0,mol_1,...,mol_n,least_similar_index,split'


def save_dataset(dataset, idx_permutations, odd_one_out_idx, split_label):
    smis = dataset["smiles"].values
    with open("data_clean.csv", "a") as f:
        for i in tqdm(range(len(smis))):
            if odd_one_out_idx[i] == np.nan:
                continue
            smi_strs = ",".join(smis[idx_permutations[i]])
            f.write(smi_strs + f",{odd_one_out_idx[i]},{split_label}\n")


if __name__ == "__main__":
    n_permutations = 4  # Controls how many molecules are given in the question

    df = load_dataset()
    save_dataset_title()

    idx_permutations, odd_one_out_idx = transform_dataset(df["train"], n_permutations)
    save_dataset(df["train"], idx_permutations, odd_one_out_idx, "train")

    idx_permutations, odd_one_out_idx = transform_dataset(df["valid"], n_permutations)
    save_dataset(df["valid"], idx_permutations, odd_one_out_idx, "valid")

    idx_permutations, odd_one_out_idx = transform_dataset(df["test"], n_permutations)
    save_dataset(df["test"], idx_permutations, odd_one_out_idx, "test")
