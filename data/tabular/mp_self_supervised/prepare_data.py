from pymatgen.core import Structure
from glob import glob
from chemnlp.data.convert import cif_file_to_string, is_longer_than_allowed

from tqdm import tqdm
import concurrent.futures
import pandas as pd

data = []


def compile_info(ciffile):
    s = Structure.from_file(ciffile)
    cif = cif_file_to_string(ciffile)
    sg, sg_n = s.get_space_group_info()

    d = {
        "formula": s.composition.reduced_formula,
        "density": s.density,
        "spacegroup": sg,
        "spacegroup_number": sg_n,
        "cif": cif,
        "is_longer_than_allowed": is_longer_than_allowed(cif),
    }
    return d


if __name__ == "__main__":
    all_structures = glob(
        "/Users/kevinmaikjablonka/LSMO Dropbox/Kevin Jablonka/small_tests/download_mp/structures/optimade_materialsproject_org/*.cif"
    )

    data = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for d in tqdm(
            executor.map(compile_info, all_structures), total=len(all_structures)
        ):
            data.append(d)

    df = pd.DataFrame(data)
    df.to_json("mpid.json")
