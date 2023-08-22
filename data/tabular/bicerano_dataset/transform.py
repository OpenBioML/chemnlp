import pandas as pd
import wget

def transform_data():


    original_data = pd.read_excel("HT_MD_polymer_properties.xlsx")
    
    clean_data = original_data.drop("sl_num", axis=1)
    assert not clean_data.duplicated().sum()

    clean_columns = ["Name", 
                     "SMILES", 
                     "Tg_exp", 
                     "Tg_calc", 
                     "Tg_calc_std", 
                     "rho_300K_exp", 
                     "rho_300K_calc",
                     "rho_300K_calc_std", 
                     "glass_CTE_calc", 
                     "glass_CTE_calc_std", 
                     "rubber_CTE_calc", 
                     "rubber_CTE_calc_std"]
    
    clean_data.columns = clean_columns
 
 
    clean_data["SMILES"] = clean_data["SMILES"].str.replace("[Ce]", "[*]", regex=False)
    clean_data["SMILES"] = clean_data["SMILES"].str.replace("[Th]", "[*]", regex=False)

    
    clean_data.to_excel("HT_MD_Polymers_clean.xlsx")

transform_data()