### This script could run a MLP-based deep neural network model comparing SEF, Morgan Fingerprint and SHED descriptors


# import the necessary packages
import numpy as np
import pandas as pd


from KiNet_mlp import KiNet_mlp

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit.Chem import AllChem

import re
import math

import scipy
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error

from tensorflow.keras.layers import Dropout


## Getting the list of molecules from 
df = pd.read_csv('CHEMBL2842_mod2.csv', encoding='cp1252')
df_target = df['Ligand Efficiency BEI/MW'].values
df_SHED = pd.read_csv('SHED_keys_CHEMBL2842_mod2.csv', encoding='cp1252')

print(df_target)
print("Shape of the Ligand Efficiency BEI/MW labels array", df_target.shape)

    
# Normalizing the target
maxPrice = df.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = df.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)


###------------------------------------------------------Shannon Entropy Generation: SMILES--------------------------------------------------------------------------------------------------

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
    # shannon_arr.append(shannon)
        
    
# print(shannon)     
df['shannon_smiles']= shannon_arr

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
  
  ps = []
  for atom_rdkit in mol.GetAtoms():
      atom_symbol = atom_rdkit.GetSymbol()
      atom_type = atom_symbol ### atom symbol in atom type
      
      partial_shannon = freq_list_input_mol[atom_type] * total_shannon
      ps.append( partial_shannon )
      # ps.append( freq_list_input_mol[atom_type] )

  ps_arr = ps_padding(ps, max_len_smiles)     
  fp_combined.append(ps_arr)

### partial or fractional shannon entropy as feature
fp_mol = pd.DataFrame(fp_combined)

### morgan fingerprint as feature
fp_morgan = pd.DataFrame(morgan_combined)

### concatenating different feature combinations of Morgan Fingerprint, SEF and SHED descriptors to feed into MLP
df_new = pd.concat([ fp_morgan, df['Ligand Efficiency BEI/MW']], axis = 1) ### only Morgan Fingerprint as descriptor

df_new = pd.concat([df['shannon_smiles'], fp_mol, df['Ligand Efficiency BEI/MW']], axis = 1)  ### Only SEF as descriptor (only Shannon entropy and fractional Shannon entropies are features)

df_new = pd.concat([df['shannon_smiles'], fp_mol, fp_morgan, df['Ligand Efficiency BEI/MW']], axis = 1)  ### both Morgan Fingerprint and SEF as descriptors

df_new = pd.concat([ df_SHED, df['Ligand Efficiency BEI/MW']], axis = 1)  ### only SHED descriptors


print("[INFO] constructing training/ testing split")
#split = train_test_split(df_new, test_size = 0.2) ### No random_state definition for Y-label randomized test
split = train_test_split(df_new, test_size = 0.2, random_state = 42) 


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

# create the MLP model
mlp = KiNet_mlp.create_mlp(XtrainData.shape[1], regress = False) # the input dimension to mlp would be shape[1] of the matrix i.e. column features
print("shape of mlp", mlp.output.shape)


# Processing output of MLP model
combinedInput = mlp.output
print("shape of combinedInput",combinedInput.shape)

# Defining the final FC layers 
x = Dense(100, activation = "relu") (combinedInput)
x = Dense(10, activation = "relu") (combinedInput)
x = Dropout(0.15)(x)   ### optional step to reduce validation loss
x = Dense(1, activation = "linear") (x)
print("shape of x", x.shape)


# Defining final model as 'model'
model = Model(inputs = mlp.input, outputs = x)
print("shape of mlp input", mlp.input.shape)


# initialize the optimizer and compile the model
print("[INFO] compiling model...")
# opt = SGD(lr= 0.95*1.0e-6, decay = 1.0e-6/200) ### for CHEMBL4691_mod2
opt = SGD(lr= 1.0e-6, decay = 1.0e-6/200)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

#train the network
print("[INFO] training network...")
trainY = XtrainLabels
testY = XtestLabels

epoch_number = 500
BS = 5;


# Defining the early stop to monitor the validation loss to avoid overfitting.
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=1, mode='auto')

# shuffle = False to reduce randomness and increase reproducibility
H = model.fit( x = XtrainData, y = trainY, validation_data = ( XtestData, testY), batch_size = BS, epochs = epoch_number, verbose=1, shuffle=False, callbacks = [early_stop]) 

# evaluate the network
print("[INFO] evaluating network...")
preds = model.predict(XtestData,batch_size=BS)

# compute the difference between the predicted and actual house prices, then compute the % difference and actual % difference
diff = preds.flatten() - testY
PercentDiff = (diff/testY)*100
absPercentDiff = (np.abs(PercentDiff)).values

# compute the mean and standard deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )

# Plotting Predicted vs Actual 
N = len(testY)
colors = np.random.rand(N)
x = testY * maxPrice
y =  (preds.flatten()) * maxPrice
plt.scatter(x, y, c=colors)
plt.plot( [0,maxPrice],[0,maxPrice] )
plt.xlabel('Actual BEI', fontsize=18)
plt.ylabel('Predicted BEI', fontsize=18)
plt.savefig('Ligand_Efficiency_BEI_pred.png')
plt.show()


####-------------------------------------------------------------------------------------------------------Evaluating standard statistics of model performance--------------------------------------------------------------------
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


# plot the training loss and validation loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoch_number ), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epoch_number ), H.history["val_loss"], label="val_loss")
plt.title("Training loss and Validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss@epochs_BEI_pred.png')
####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------