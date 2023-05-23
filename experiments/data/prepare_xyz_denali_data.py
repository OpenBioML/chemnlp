import glob
import json
import re
from typing import Dict, List

import joblib
from tqdm import tqdm

FLOATS = r"\b\d+\.\d+\b"
N_DIGITS = 2
DATA_DIR = "/fsx/proj-chemnlp/data/OrbNet_Denali_Training_Data/xyz_files"
OUT_DIR = "/fsx/proj-chemnlp/data/cleaned_orbnet_xyz"


def round_to_n(match: str) -> str:
    """Rounds a number to N_DIGIT places"""
    num = float(match.group(0))
    return f"{round(num, N_DIGITS)}"


def process_xyz_file(fpath: str) -> Dict:
    """
    Converts an xyz coordinate txt file

    from:

    0 1
    C     6.387857024727   2.835389827222  -1.459309052282
    O     6.124745792088   1.572472127721  -0.856663518011
    P     5.135493394273   1.100298055313   0.304913348284
    O     4.810980512745   2.057739129140   1.356652377607
    N     3.815309502285   0.536342942394  -0.613493651219
    C     2.556034965626   0.186837010612  -0.140419247288
    C     2.305870863093   0.023851551509   1.222745502583
    C     1.037977807704  -0.316893348828   1.657755448886
    C     0.004741967570  -0.509277258465   0.739473499249
    N    -1.259660201010  -0.892365146475   1.228495785578
    ................................................. more

    to:

    "C 6.39 2.84 -1.46 O 6.12 1.57 -0.86 P 5.14 1.10 0.31 ..."
    """
    with open(fpath, "r") as f:
        var = f.read()  # read in text file
        lines = var.split("\n")  # split on newline
        xyz_s = " ".join(lines[2:])  # removed first two items
        xyz_onespace = " ".join(xyz_s.split())  # make uniform spacing
        rounded_xyz = re.sub(FLOATS, round_to_n, xyz_onespace)  # round numbers
        return {"text": rounded_xyz, "num_atoms": int(len(rounded_xyz) / 4)}


def save_jsonlines(out_dir: str, data_json_clean: List[Dict]):
    """Saves down to jsonlines format"""
    with open(f"{out_dir}/data_clean.jsonl", "w") as file_out:
        # this could be parallelised too
        for element in data_json_clean:
            json.dump(element, file_out)
            file_out.write("\n")


def get_xyz_files(base_dir: str) -> List[str]:
    """Gets all matches to .xyz extension"""
    return glob.iglob(f"{base_dir}/**/*.xyz", recursive=True)


if __name__ == "__main__":
    examples = get_xyz_files(DATA_DIR)
    print("Retrieved generator of all xyz files, processing now ...")
    data_json_clean = joblib.Parallel(n_jobs=joblib.cpu_count())(
        [joblib.delayed(process_xyz_file)(fpath) for fpath in tqdm(examples)]
    )
    print(f"Processed all {len(data_json_clean)} xyz files")
    save_jsonlines(OUT_DIR, data_json_clean)
    print(f"Saved processed files to {OUT_DIR}")
