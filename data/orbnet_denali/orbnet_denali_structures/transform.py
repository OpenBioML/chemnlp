from chemnlp.utils import xyz_to_mol, load_config, extract_tarball
import os 
from glob import glob
import pandas as pd 
from rdkit import Chem

_THIS_DIR =  os.path.dirname(os.path.abspath(__file__))
_LABEL_DIR = os.path.join(_THIS_DIR,'../labels')
_STRUCTURE_DIR = os.path.join(_THIS_DIR, '../xyz_files')


def _download_files_if_not_existent(config):
    xyz_files = glob(os.path.join(_STRUCTURE_DIR, '**', "*.xyz"))
    if not len(xyz_files) == config['num_points']:
        print(f"🚧 Didn't find all xyz files ..., found {len(xyz_files)}, expected {config['num_points']}")
        structure_link = list(filter(lambda x: x['description'] == 'structure download', config['links']))
        extract_tarball(structure_link[0]['url'], _STRUCTURE_DIR)

    csv_file = os.path.join(_LABEL_DIR, 'denali_labels.csv')
    if not os.path.exists(csv_file) or len(pd.read_csv(csv_file)) != config['num_points']:
        print("🚧 Didn't find enough rows")
        csv_link = list(filter(lambda x: x['description'] == 'label download', config['links']))
        extract_tarball(csv_link[0]['url'], _LABEL_DIR)


def _add_smiles_and_filepath(row):
    filepath = os.path.join(_STRUCTURE_DIR, row['mol_id'], row['sample_id'] + '.xyz')
    mol = xyz_to_mol(filepath, row['charge'])
    path = os.path.abspath(filepath)

    row['SMILES'] = Chem.MolToSmiles(mol)
    row['path'] = str(path)

    return row

def _create_smiles_and_filepath():
    df = pd.read_csv(os.path.join(_LABEL_DIR, 'denali_labels.csv'))

    df.apply(_add_smiles_and_filepath, axis=1)

    df.dropna(subset=['SMILES', 'path', 'charge'], inplace=True)

    df.to_csv('data_clean.csv')

if __name__ == '__main__':
   config = load_config(os.path.join(_THIS_DIR, 'meta.yaml'))

   _download_files_if_not_existent(config=config)

   _create_smiles_and_filepath()
