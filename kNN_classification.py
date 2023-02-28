#### This script runs kNN-based machine learning classification model based on ECFP4-based Tanimoto similarity between molecules


# import the necessary packages
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

import time
time_start = time.time()


### Getting the list of molecules 

### for ames mutagenicity data
df =  pd.read_csv('features_mutagenicity_with_shannon_with_smiles.csv', encoding='cp1252') 
df_smiles = df.iloc[:,2085].values
df_target = df.iloc[:,-1].values
    
# Normalizing factor for the target
maxPrice = df.iloc[:,-1].max() # grab the maximum price in the training set's last column
minPrice = df.iloc[:,-1].min() # grab the minimum price in the training set's last column
print(maxPrice,minPrice)

X = df_smiles
y = df_target



print("[INFO] constructing training/ testing split")
split = train_test_split(X,y,shuffle = True, test_size = 0.15, random_state = 10) ### for ames_mutagenicity data

### Distribute the input data columns in train & test splits    
(Xtrain_Data, Xtest_Data, ytrain_Data, ytest_Data) = split  # split format always is in (data_train, data_test, label_train, label_test)

### normalizing the target (optional for ames mutagenicity case)
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
            
            if pred>0.5:
                
                y_predict.append(1)
                
            else:
                y_predict.append(0)
                   
            
        return y_predict
    

### Running k = 1,2,3 manually to get optimum performance      
knn_model = KNN_regressor(k=3)  

knn_model.fit(XtrainData,XtrainLabels)        
    
preds = knn_model.predict(XtestData)    
    
    
###-------------------------------------------------------------------Statistics----------------------------------------------------------------------------------------------------------------------------------------------------    
print(classification_report (XtestLabels, preds, target_names = ["non-mutagenic" , "mutagenic"] ))    

# evaluating the confusion matrix
conf_mx = confusion_matrix(XtestLabels, preds)
plt.matshow(conf_mx, cmap = 'binary')
plt.show()
# ####-------------------------------------------------------------------------------------------------------Evaluating standard statistics of model performance--------------------------------------------------------------------
# Estimating the AUC
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(XtestLabels, preds)
roc_auc = metrics.auc(fpr, tpr)

# plotting the ROC_AUC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('AUC_ROC_with_all_descriptors_partial_shannon')
plt.show()

# ####------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

time_end = time.time()
print("Time taken for simulation: ", time_end - time_start)