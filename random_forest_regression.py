### This script runs RFR-based machine learning models based on SEF, Morgan fingerprints and SHED descriptors


# import the necessary packages

import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import MinMaxScaler


from rdkit import Chem
from rdkit.Chem import AllChem

import re
import math

import scipy
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error

import time
time_start = time.time()
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### getting data from .csv file
### CHEMBL2842
df = pd.read_csv('CHEMBL2842_mod2.csv', encoding='cp1252') 
print("shape of data", np.shape(df))
df_target = df['Ligand Efficiency BEI/MW'].values
print("Shape of the Ligand Efficiency BEI labels array", df_target.shape)
df_SHED = pd.read_csv('SHED_keys_CHEMBL2842_mod2.csv', encoding='cp1252')
    
# Normalizing factor for the target
maxPrice = df.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = df.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###------------------------------------------------------Shannon Entropy Generation: SMILES/ SMARTS/ InChiKey-----------------------------------------------------------------------------------------------

### Generate a new column with title 'shannon_smiles'. Evaluate the Shannon entropy for each smile string and store into 'shannon_smiles' column

### Inserting the new column as the 2nd column in df
df.insert(1, "shannon_smiles", 0.0)

# smiles regex definition
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)

shannon_arr = []
for p in range(0,len(df['shannon_smiles'])):
    
    molecule = df['Smiles'][p]
    tokens = regex.findall(molecule)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)
        
    # print("shannon entropy: ", shannon)
    shannon_arr.append(shannon)
    
    # shannon_arr.append(math.exp(-shannon))
        
    
# print(shannon)     
df['shannon_smiles']= shannon_arr


df.insert(2, "shannon_smarts", 0.0)
df.insert(3, "shannon_inchikey", 0.0)

### smarts
smarts_REGEX_PATTERN =  r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(smarts_REGEX_PATTERN)


shannon_arr = []
for p in range(0,len(df['shannon_smarts'])):
    
    smiles = df['Smiles'][p]
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)
    m_smarts = Chem.MolToSmarts(mol)
    
    molecule = m_smarts
    tokens = regex.findall(molecule)
    # print(tokens)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)
        
    # print("shannon entropy on smarts: ", shannon)
    shannon_arr.append(shannon)  
    
df['shannon_smarts']= shannon_arr        


### inchikey
### InChiKey: reading all letters except special characters
InChiKey_REGEX_PATTERN = r"""([A-Z])"""
regex = re.compile(InChiKey_REGEX_PATTERN)

def shannon_entropy_inch(ik):
    
    molecule = ik
    tokens = regex.findall(molecule)
    # print(tokens)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)
        
    return shannon  

shannon_arr = []
for p in range(0,len(df['shannon_inchikey'])):

    smiles = df['Smiles'][p]
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)
    m_inchkey = Chem.MolToInchiKey(mol)
    ik = m_inchkey.split("-")
    # print("InchiKey splitted", ik)
    # print("\n")

    shannon_entropy = 0
    for i in range(len(ik)):
    
        if i<=1:
            shannon_entropy_by_parts = shannon_entropy_inch(ik[i])
            shannon_entropy = shannon_entropy + shannon_entropy_by_parts  
            # print(shannon_entropy_by_parts)
        else:
            freq = 1/25 ### Inchikey contains total 25 characters, apart from 2 hyphens
            shannon_entropy_by_parts = - freq * math.log2(freq)
            shannon_entropy = shannon_entropy + shannon_entropy_by_parts  
            # print(shannon_entropy_by_parts)
        
        
    # print("shannon entropy on inchikey: ", shannon_entropy)
    shannon_arr.append(shannon_entropy) 
    
df['shannon_inchikey']= shannon_arr 
###-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
   
###---------------------------------------------------------------------SMILES Shannon estimated in function form-------------------------------------------------------------------------------------------------------------------------

# smiles regex pattern
SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
regex = re.compile(SMI_REGEX_PATTERN)
def shannon_entropy_smiles(mol_smiles):
    
    molecule = mol_smiles 
    tokens = regex.findall(molecule)
    
    ### Frequency of each token generated
    L = len(tokens)
    L_copy = L
    tokens_copy = tokens
    
    num_token = []
    
    
    for i in range(0,L_copy):
        
        token_search = tokens_copy[0]
        num_token_search = 0
        
        if len(tokens_copy) > 0:
            for j in range(0,L_copy):
                if token_search == tokens_copy[j]:
                    # print(token_search)
                    num_token_search += 1
            # print(tokens_copy)        
                    
            num_token.append(num_token_search)   
                
            while token_search in tokens_copy:
                    
                tokens_copy.remove(token_search)
                    
            L_copy = L_copy - num_token_search
            
            if L_copy == 0:
                break
        else:
            pass
        
    # print(num_token)
    
    ### Calculation of Shannon entropy
    total_tokens = sum(num_token)
    
    shannon = 0
    
    for k in range(0,len(num_token)):
        
        pi = num_token[k]/total_tokens
        
        # print(num_token[k])
        # print(math.log2(pi))
        
        shannon = shannon - pi * math.log2(pi)
    
    # shannon = math.exp(-shannon)    
        
    return shannon   
###---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  

### generating a dictionary of atom occurrence frequencies with the following atoms in the database: 'H', 'C', 'N', 'O', 'S','F','Cl'
def freq_atom_list(atom_list_input_mol):
    
    
    atom_list = ['H', 'C', 'N', 'O', 'S','F','Cl']  ### For CHEMBL2842_mod2
    
    
    dict_freq = {}
    
    ### adding keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = 0  ### The values are all set 0 initially
    # print(dict_freq)
    
    ### update the value by 1 when a key in encountered in the string
    for i in range(len(atom_list_input_mol)):
        dict_freq[ atom_list_input_mol[i] ] = dict_freq[ atom_list_input_mol[i] ] + 1
    
    ### The dictionary values as frequency array
    freq_atom_list =  list(dict_freq.values())/ (  sum(  np.asarray (list(dict_freq.values()))  )    )
    
    # print(list(dict_freq.values()))
    # print(freq_atom_list )
    
    ### Getting the final frequency dictionary
    ### adding values to keys
    for i in range(len(atom_list)):
        dict_freq[atom_list[i]] = freq_atom_list[i]  
        
    # print(dict_freq)
    freq_atom_list = dict_freq
    
        
    return freq_atom_list


### generating a dictionary of bond occurrence frequencies with the following bond types: 'SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'AROMATIC', 'HYDROGEN', 'IONIC'
def freq_bond_list(bond_list_input_mol):
    
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'QUADRUPLE', 'AROMATIC', 'HYDROGEN', 'IONIC'] 
    dict_freq = {}
    
    ### adding keys
    for i in range(len(bond_list)):
        dict_freq[bond_list[i]] = 0  ### The values are all set 0 initially
    # print(dict_freq)
    
    ### update the value by 1 when a key in encountered in the string
    for i in range(len(bond_list_input_mol)):
        dict_freq[ bond_list_input_mol[i] ] = dict_freq[ bond_list_input_mol[i] ] + 1
    
    ### The dictionary values as frequency array
    freq_bond_list =  list(dict_freq.values())/ (  sum(  np.asarray (list(dict_freq.values()))  )    )
    
    # print(list(dict_freq.values()))
    # print(freq_bond_list )
    
    ### Getting the final frequency dictionary
    ### adding values to keys
    for i in range(len(bond_list)):
        dict_freq[bond_list[i]] = freq_bond_list[i]  
        
    # print(dict_freq)
    freq_bond_list = dict_freq
    
        
    return freq_bond_list


### Maximum length of SMILES strings in the database
len_smiles = []
for j in range(0,len(df['Smiles'])):
    
    mol = Chem.MolFromSmiles(df['Smiles'][j])
    
    ### No H atom considered
    # mol = Chem.AddHs(mol)
    
    k=0
    for atom in mol.GetAtoms():
        k = k +1 
    len_smiles.append(k)

max_len_smiles = max(len_smiles)


### Constructing the padded array of partial or fractional shannon per molecule
def ps_padding(ps, max_len_smiles):
    
    len_ps = len(ps)
    
    len_forward_padding = int((max_len_smiles - len_ps)/2)
    len_back_padding = max_len_smiles - len_forward_padding - len_ps
    
    ps_padded = list(np.zeros(len_forward_padding))  + list(ps) + list(np.zeros(len_back_padding))
    
    return ps_padded 


### collecting the features to use as descriptor array

fp_combined = []
morgan_combined = []

fp_bond = []
fp_bond_shannon = []

for i in range(0,len(df['Smiles'])):  
    
  mol = Chem.MolFromSmiles(df['Smiles'][i])
  ### estimating the partial shannon for an atom type => the current node
  total_shannon = shannon_entropy_smiles(df['Smiles'][i])
  # shannon_arr.append( total_shannon )
  
  ### Estimating Morgan Fingerprints
  info = {}
  fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits = 1024, bitInfo=info)
  vector = np.array(fp_morgan)
  morgan_combined.append(vector)
  
  ### The atom list as per rdkit in string form
  atom_list_input_mol = []
  for atom_rdkit in mol.GetAtoms():
     atom_list_input_mol.append(str(atom_rdkit.GetSymbol()))     
        
     
  freq_list_input_mol = freq_atom_list(atom_list_input_mol)
  
  
  ### The bond list as per rdkit in string form
  bond_list_input_mol = []
  for i in range(len(mol.GetBonds())):
      bond_list_input_mol.append(str(mol.GetBonds() [i].GetBondType()))
    
  ### bond frequency dictionary  
  freq_list_input_mol_bond = freq_bond_list(bond_list_input_mol)  
  
  ps = []
  for atom_rdkit in mol.GetAtoms():
      atom_symbol = atom_rdkit.GetSymbol()
      atom_type = atom_symbol ### atom symbol in atom type
      
      partial_shannon = freq_list_input_mol[atom_type] * total_shannon
      ps.append( partial_shannon )
      # ps.append( freq_list_input_mol[atom_type] )
      
  
        
  ps_arr = ps_padding(ps, max_len_smiles)     
  fp_combined.append(ps_arr)    
  
  bond_arr = freq_list_input_mol_bond.values()
  fp_bond.append(bond_arr)
    
  ### converting dictionary array into a list array
  bond_list = [i for i in bond_arr]
    
  # estimation of total bond entropy (bond_shannon)
  bond_shannon = 0
    
  for k in range(0,len(bond_list)):
        
      pi = bond_list[k]
        
      # print(num_token[k])
      # print(math.log2(pi))
      if pi>0:
            bond_shannon = bond_shannon - pi * math.log2(pi)
         
  fp_bond_shannon.append(bond_shannon)    
      


### partial or fractional shannon entropy as feature
fp_mol = pd.DataFrame(fp_combined)


# bond frequency as feature or descriptor column
freq_bond = pd.DataFrame(fp_bond)

# bond Shannon entropy as feature or descriptor column
bond_shannon = pd.DataFrame(fp_bond_shannon)

### morgan fingerprint as feature
fp_morgan = pd.DataFrame(morgan_combined)


### concatenating different feature combinations to feed into RFR model

### CHEMBL2842
df_new = pd.concat([ fp_morgan, df['Ligand Efficiency BEI/MW']], axis = 1) ### for only Morgan
# ### df_new = pd.concat([df['shannon_smiles'], fp_mol, df['Ligand Efficiency BEI/MW']], axis = 1)  ### for only SEF
# df_new = pd.concat([df['shannon_smiles'], fp_mol, freq_bond, df['Ligand Efficiency BEI/MW']], axis = 1)  ### for only SEF (more efficient)
# df_new = pd.concat([df['shannon_smiles'], fp_mol, freq_bond, fp_morgan, df['Ligand Efficiency BEI/MW']], axis = 1)  ### for Morgan + SEF
#df_new = pd.concat([ df_SHED, df['Ligand Efficiency BEI/MW']], axis = 1) ### for SHED
# #df_new = pd.concat([ df['shannon_smiles'], fp_mol, freq_bond, df_SHED, df['Ligand Efficiency BEI/MW']], axis = 1) ### for SHED + SEF
#---------------------------------------------------------------------------------------------------------------------------------------------------

print("[INFO] constructing training/ testing split")
###split = train_test_split(df_new, test_size = 0.1, random_state = 10) 
split = train_test_split(df_new, test_size = 0.2, shuffle = True, random_state = 42) 
# split = train_test_split(df_new, test_size = 0.2, shuffle = True)  ### Y-target randomized

# Distribute the input data columns in train & test splits    
(XtrainTotalData, XtestTotalData) = split  # split format always is in (data_train, data_test, label_train, label_test)

# ## Taking only the (Ligand Efficiency BEI/MW) as the target
maxPrice = df_new.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = df_new.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)

XtrainLabels  = (XtrainTotalData.iloc[:,-1])/ (maxPrice)
XtestLabels  = (XtestTotalData.iloc[:,-1])/ (maxPrice)    

# All columns except the last as X data
XtrainData = (XtrainTotalData.iloc[:,0:-1])
XtestData = (XtestTotalData.iloc[:,0:-1])

print("XtrainData shape",XtrainData.shape)
print("XtestData shape",XtestData.shape)


# perform min-max scaling each continuous feature column to the range [0 1]
cs = MinMaxScaler()
trainContinuous = cs.fit_transform(XtrainData)
testContinuous = cs.transform(XtestData)

print("[INFO] processing input data after normalization....")
XtrainData, XtestData = trainContinuous,testContinuous ## Feeding single array as XtrainData and XtestData


# Rename the data & targets
X_train,X_test,y_train,y_test = XtrainData, XtestData, XtrainLabels, XtestLabels


# Define model with GridSearch
RFR_model = RandomForestRegressor()


param_grid = { 
            "n_estimators"    : [25,100,200],
            "min_samples_leaf"      : [1,2,5],
            "min_samples_split" : [2,3,5],
             }

cv_fold = 5
opt_metric = "neg_mean_absolute_error"


cv_results = GridSearchCV(RFR_model, param_grid = param_grid, cv=cv_fold, scoring=opt_metric, refit = True)
cv_results.fit(X_train,y_train)

# get optimal parameters
best_params = cv_results.best_params_


time_end_opt = time.time()
print("Time taken for model parameter optimization: ", time_end_opt - time_start)


# define model again with optimal parameters
RFR_model = cv_results.best_estimator_
print("The optimized RFR model: ", RFR_model)

#Fit the optimal model
RFR_model.fit(X_train,y_train)

#Evaluate model
regressor_prediction = RFR_model.predict(X_test)

# Plot the predicted data with correlation
correlation = round(pearsonr(regressor_prediction,y_test)[0],5)
print("The correlation between predicted and input y-values", correlation)

output_filename = "RFR_Regression.png"
title_name = " Regression by RFR- Real vs Predicted"
x_axis_label = "Real"
y_axis_label = "Predicted"

# Defining a plotting function
def simple_scatter_plot(y_test,regressor_prediction,output_filename,title_name,x_axis_label,y_axis_label):
    
    seaborn.set(color_codes = True)
    plt.figure(1, figsize = (9,6))
    plt.title(title_name)
    axis = seaborn.scatterplot(x = y_test* maxPrice, y = regressor_prediction* maxPrice )
    axis.set(xlabel = x_axis_label, ylabel = y_axis_label )
    plt.savefig(output_filename, bbox_inches = 'tight', dpi = 300)
    plt.close()

# PLot the correlaton graph
simple_scatter_plot(y_test*maxPrice,regressor_prediction*maxPrice,output_filename,title_name,x_axis_label,y_axis_label)

#-----------------------------------------------------------------------Statistics--------------------------------------------------------------------------------------

# # Plotting Predicted vs Actual 
N = len(y_test)
colors = np.random.rand(N)
x = y_test * maxPrice
y = regressor_prediction * maxPrice
plt.scatter(x, y, c=colors)
plt.plot( [0,maxPrice],[0,maxPrice] )
plt.xlabel('Actual', fontsize=18)
plt.ylabel('Predicted', fontsize=18)
plt.savefig('Ligand_Efficiency_BEI_pred.png')
plt.show()

# compute the difference between the predicted and actual house prices, then compute the % difference and actual % difference
diff = x - y
PercentDiff = (diff/x)*100
absPercentDiff = (np.abs(PercentDiff)).values

# compute the mean and standard deviation of absolute percentage difference (MAPE)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )


# ####-------------------------------------------------------------------------------------------------------Evaluating standard statistics of model performance--------------------------------------------------------------------
# ### MAE as a function
# def mae(y_true, predictions):
#     y_true, predictions = np.array(y_true), np.array(predictions)
#     return np.mean(np.abs(y_true - predictions))

# ### The MAE estimated
# print("The mean absolute error estimated: {}".format( mae(x, y) )) 

### The MAPE
print("The mean absolute percentage error: {}".format( mean_absolute_percentage_error(x, y) ) )   

### The MAE
print("The mean absolute error: {}".format( mean_absolute_error(x, y) ))    
    
### The MSE
print("The mean squared error: {}".format( mean_squared_error(x, y) ))  

### The RMSE
print("The root mean squared error: {}".format( mean_squared_error(x, y, squared= False) ) )    

### General stats
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print("The R^2 value between actual and predicted target:", r_value**2)

# #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


time_end = time.time()
print("Time taken for simulation: ", time_end - time_start)