### This script could build SHED keys from list of SMILES


# import the necessary packages
from imutils import paths
import random
import os
import numpy as np
import pandas as pd


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools


import subprocess
import shlex

import shutil
import os


import time


start_time = time.time()

###-----------------------------------------------reading the data----------------------------------------------------------------------------------
# Getting the list of molecules from csv file obtained from ChEMBL database
df = pd.read_csv('CHEMBL2842_mod2.csv', encoding='cp1252') 
df_smiles= df['Smiles'].values 
print("Shape of the ligand array", df_smiles.shape)

SHED = []
for i in range(0,len(df_smiles)):
    
    smiles = df_smiles[i]
    mol = Chem.MolFromSmiles(df_smiles[i])
    mol = Chem.AddHs(mol)
    
    w = Chem.SDWriter('{}th_with_H.sdf'.format(i)) ### one molecule per sdf. Need to use w.flush() along with this
    
    AllChem.Compute2DCoords(mol)
    w.write(mol)
    
    
    w.flush() 
     

for i in range(0,len(df_smiles)):

    cnt = i
    ### Accessing java (jre) command line to get SHED keys
    cmd_line_jcmapper_SHED = 'java -jar jCMapperCLI.jar -f {k}th_with_H.sdf -c SHED -ff FULL_TAB_UNFOLDED -o {k}th_with_H.csv'.format(k=cnt)
    args = shlex.split(cmd_line_jcmapper_SHED)
    txt = subprocess.check_output(args)

    df_SHED = pd.read_csv('{}th_with_H.csv'.format(i), encoding='cp1252', delimiter = '\t', header= None) 
    shed_keys = df_SHED[2]
    
    SHED.append(shed_keys)

df_SHED_descriptor = pd.DataFrame(SHED)   


### Creating new folders
newpath1 = r'sdf_files' 
newpath2 = r'jcMapper_csv_files' 
if not os.path.exists(newpath1) and not os.path.exists(newpath2):
    os.makedirs(newpath1)
    os.makedirs(newpath2)

for i in range(0,len(df_smiles)):

    try:    
        shutil.move("{}th_with_H.sdf".format(i) , "sdf_files/{}th_with_H.sdf".format(i))
               
        shutil.move("{}th_with_H.csv".format(i) , "jcMapper_csv_files/{}th_with_H.csv".format(i))
        
    except:
        continue


### Saving the SHED features in .csv file format
df_SHED_descriptor.to_csv('SHED_keys_CHEMBL2842_mod2.csv', index=False)


end_time = time.time()
processing_time = end_time - start_time

print("Data processing time: ", processing_time)


