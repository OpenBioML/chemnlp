import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
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
    similarity_list = []

    most_diff_pairs = []
    biggest_sim_pairs = []
    similarity_sum_list = []

    smallest_similariies = []
    biggest_similarities = []
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
            similarity_sums = []
            most_diff = None

            smallest_similarity = np.inf

            biggest_sim = None
            biggest_similarity = -np.inf

            for i in range(len(fingerprints)):
                similarity_sum = 0
                for j in range(len(fingerprints)):
                    if i != j:
                        sim = DataStructs.TanimotoSimilarity(
                            fingerprints[i], fingerprints[j]
                        )
                        similarity_sum += sim
                        if sim < smallest_similarity:
                            smallest_similarity = sim
                            most_diff = (i, j)
                        if sim > biggest_similarity:
                            biggest_similarity = sim
                            biggest_sim = (i, j)
                similarity_sums.append(similarity_sum)
                similarities.append(sim)

            most_diff_pairs.append(most_diff)
            biggest_sim_pairs.append(biggest_sim)
            smallest_similariies.append(smallest_similarity)
            biggest_similarities.append(biggest_similarity)

            lowest_similarity = np.argmin(similarity_sums)
            odd_one_out_idx.append(lowest_similarity)
            similarity_list.append(similarities)
            similarity_sum_list.append(similarity_sums)
        except Exception:
            odd_one_out_idx.append(np.nan)
            similarity_list.append(np.nan)
            biggest_sim_pairs.append(np.nan)
            lowest_similarity = np.nan
            smallest_similariies.append(np.nan)
            biggest_similarities.append(np.nan)
            similarity_sum_list.append(np.nan)
            most_diff_pairs.append(np.nan)

    return {
        # "smi_idx_arr": smi_idx_arr,
        "smi_1": smis[smi_idx_arr[:, 0]],
        "smi_2": smis[smi_idx_arr[:, 1]],
        "smi_3": smis[smi_idx_arr[:, 2]],
        "smi_4": smis[smi_idx_arr[:, 3]],
        "odd_one_out_idx": odd_one_out_idx,
        "odd_one_out_mol": [
            (
                smis[smi_idx_arr[i, int(odd_one_out_idx[i])]]
                if not np.isnan(odd_one_out_idx[i])
                else np.nan
            )
            for i in range(len(odd_one_out_idx))
        ],
        # "similarity_list": similarity_list,
        "smallest_to_second_smallest_ratio": [
            division_catch_zero(x) for x in similarity_sum_list
        ],
        "most_diff_0": [
            smis[smi_idx_arr[i, x[0]]] if not isinstance(x, float) else np.nan
            for i, x in enumerate(most_diff_pairs)
        ],
        "most_diff_1": [
            smis[smi_idx_arr[i, x[1]]] if not isinstance(x, float) else np.nan
            for i, x in enumerate(most_diff_pairs)
        ],
        "biggest_sim_0": [
            smis[smi_idx_arr[i, x[0]]] if not isinstance(x, float) else np.nan
            for i, x in enumerate(biggest_sim_pairs)
        ],
        "biggest_sim_1": [
            smis[smi_idx_arr[i, x[1]]] if not isinstance(x, float) else np.nan
            for i, x in enumerate(biggest_sim_pairs)
        ],
        "smallest_similarities": smallest_similariies,
        "biggest_similarities": biggest_similarities,
    }


def division_catch_zero(x):
    try:
        x = sorted(x)
        try:
            return x[0] / x[1]
        except ZeroDivisionError:
            return np.nan
    except Exception:
        return np.nan


# tune this to tune difficulty of task, not optimized atm
MAX_SECOND_FIRST_RATIO = 0.5
MIN_SMALLEST_TO_LARGEST_DIFF = 0.2

if __name__ == "__main__":
    n_permutations = 4  # Controls how many molecules are given in the question

    df = load_dataset()

    out_train = pd.DataFrame(transform_dataset(df["train"], n_permutations))

    out_valid = pd.DataFrame(transform_dataset(df["valid"], n_permutations))

    out_test = pd.DataFrame(transform_dataset(df["test"], n_permutations))

    all_data = pd.concat([out_train, out_valid, out_test])
    all_data["smallest_to_largest_diff"] = (
        all_data["biggest_similarities"] - all_data["smallest_similarities"]
    )
    all_data = all_data[
        all_data["smallest_to_largest_diff"] >= MIN_SMALLEST_TO_LARGEST_DIFF
    ]
    all_data = all_data[
        all_data["smallest_to_second_smallest_ratio"] <= MAX_SECOND_FIRST_RATIO
    ]
    all_data.dropna(inplace=True)
    print(len(all_data))
    all_data.to_csv("data_clean.csv", index=False)
