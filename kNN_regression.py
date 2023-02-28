### This script runs kNN-based machine learning model based on ECFP4-based Tanimoto similarity between molecules


# import the necessary packages
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

import scipy
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error

from sklearn.model_selection import GridSearchCV

import time

time_start = time.time()


### Getting the list of molecules from a .csv file
df = pd.read_csv('CHEMBL2842_mod2.csv', encoding='cp1252') 
df_target = df['Ligand Efficiency BEI/MW'].values

    
### Normalizing factor for the target
maxPrice = df.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = df.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)

X = df['Smiles'].values
y = df_target


print("[INFO] constructing training/ testing split")
# split = train_test_split(X, y, test_size = 0.2, shuffle = True) ### random X and y split 
split = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 42) 


### Distribute the input data columns in train & test splits    
(Xtrain_Data, Xtest_Data, ytrain_Data, ytest_Data) = split  # split format always is in (data_train, data_test, label_train, label_test)

### Taking only the (Ligand Efficiency BEI/MW) as the target
maxPrice = y.max() # grab the maximum price in the training set's last column
minPrice = y.min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)

XtrainLabels  = ytrain_Data/(maxPrice)
XtestLabels  = ytest_Data/(maxPrice)    

### All columns except the last as X data
XtrainData = Xtrain_Data
XtestData = Xtest_Data

print("XtrainData shape",XtrainData.shape)
print("XtestData shape",XtestData.shape)


###---------------------------------------------------------------------KNN algo -----------------------------------------------------------------------
    
###-----------------------------------------------------------------------------------------------------------------------------------------------------

    
class KNN_regressor():
    
    def __init__(self, k):
        self.k = k


    # This function is used for training: defining the fit function definition to create data instances
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
        # print(x_train)
        # print(y_train)
        
    # This function estimates the similarity score between the test and train data points    
    def similarity_metric(self,tr_x_p,te_x_p,*kwargs):
    
        
        # print("The training X data: ", tr_x_p)
        # print("The testing X data: ", te_x_p)
    
        mol_train = Chem.MolFromSmiles(tr_x_p)

    
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_train, 2, 2048)
        
        
        mol_test = Chem.MolFromSmiles(te_x_p)

        
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_test, 2, 2048)


        Tanimoto_coeff = DataStructs.TanimotoSimilarity(fp1,fp2)
        
        # print("Tanimoto_coeff: ", Tanimoto_coeff)
        
        return Tanimoto_coeff 
    

    # This function runs the k nearest neighbour algorithm and
    # returns the predicted label/ target with the largest similarity
    def predict(self, x_test):
        
        y_predict = []

        for i in range(len(x_test)):
            sim = []
            index = []
            
            # print("The x_test[i]: ", x_test[i])
            
            for j in range(self.x_train.shape[0]):
                
                # print("The x_train[j]: ", self.x_train[j])
                # print("The y_train: ",self.y_train[j])
                sim_val = self.similarity_metric( self.x_train[j] , x_test[i] )
                
                sim.append(sim_val)
                
            sim = np.array(sim)    
            
            # sorting indexes as per high to low values
            index = np.argsort(sim)[::-1] 
            
            # getting the sorted target array
            y_training_sorted = self.y_train[index]
            
            # prediction is the first k average values from sorted training targets
            pred = np.average(y_training_sorted[:self.k])
            
            y_predict.append(pred)
            
        return y_predict
    

### Running k = 1,2,3 manually to get optimum performance  
  
# knn_model = KNN_regressor(k=2)  
knn_model = KNN_regressor(k=1) 
# knn_model = KNN_regressor(k=3) 

knn_model.fit(XtrainData,XtrainLabels)        
    
preds = knn_model.predict(XtestData)    
    
    
###-------------------------------------------------------------------Statistics-------------------------------------------------------------------------    
    
# compute the difference between the predicted and actual house prices, then compute the % difference and actual % difference
diff = preds - XtestLabels
PercentDiff = (diff/XtestLabels)*100
absPercentDiff = (np.abs(PercentDiff))

# compute the mean and standard deviation of absolute percentage difference (MAPE)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("[INF0] mean {: .2f},  std {: .2f}".format(mean, std) )

# Plotting Predicted vs Actual 
N = len(XtestLabels)
colors = np.random.rand(N)
x = XtestLabels * maxPrice
y =  [preds[i] * maxPrice for i in range(len(preds))]
plt.scatter(x, y, c=colors)
plt.plot( [0,maxPrice],[0,maxPrice] )
plt.xlabel('Actual', fontsize=18)
plt.ylabel('Predicted ', fontsize=18)
plt.savefig('Ligand_Efficiency_BEI_pred_knn.png')
plt.show()


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

# ####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

time_end = time.time()
print("Time taken for simulation: ", time_end - time_start)