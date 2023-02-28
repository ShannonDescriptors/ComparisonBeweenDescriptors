# ComparisonOfDescriptors
Comparison between Shannon Entropy Framework-based Descriptors (SEF), Morgan Fingerprint and SHED descriptors

Harnessing Shannon entropy of molecular symbols in deep neural networks to enhance prediction accuracy
------------------------------------------------------------------------------------------------------
This repository holds the codes pertaining to Tables 2 and 3 of the article 'Harnessing Shannon entropy of molecular symbols in deep neural networks to enhance prediction accuracy'.

Description
-----------
Shannon entropy framework has been demonstrated as an efficient descriptor for regression-type machine learning problem using MLP-based deep neural networks as well as general machine learning models. In this specific case, we scale up the Shannon entropy-based descriptors (SEF) to model and predict the following: (i) BEI (Ligand Binding Efficiency Index), pChEMBL and LogP values of molecules or ligands of different target IDs from ChEMBL, (ii) provide kNN baseline model predictions of the respective datasets corresponding to the target IDs and (iii) compare applicability and performance of Morgan Fingerprint, SEF, SHED descriptors and their combinations across MLP-based deep neural networks and random forest-based general machine learning models. The specific objectives of the codes are described in the Notes section below. The basic dataset has been provided in the repository in the form of .csv files.

Usage
-----
1. Download or make a clone of the repository
2. Make a new conda environment using the environment file 'mlp_dnn.yml' as provided in other sister repositories
3. Run the python files directly using a python IDE or from command line

Example: python random_forest_regression.py

Notes
-----
1. The function file is KiNet_mlp.py. Therefore, directly run the other python files apart from this one.

2. The provided .csv files are: (i) downloaded and curated dataset for the BEI model/prediction of the target ID CHEMBL2842: 'CHEMBL2842_mod2.csv', (ii) generated SHED descriptors for the BEI model/prediction of target ID CHEMBL2842 using the python script file 'data_prep_SHED.py': 'SHED_keys_CHEMBL2842_mod2.csv'

3. The jre executable 'jCMapperCLI.jar' is required to generate SHED descriptors running the 'data_prep_SHED.py'

4. The objectives and usage of the rest of the scripts are as follows: Please run the python scripts directly or using the command line 'python <script_name.py> from the terminal

(i) kNN_regression.py: This script runs kNN-based machine learning regression models based on ECFP4-based Tanimoto similarity between molecules.

(ii) kNN_classification.py: This script runs kNN-based machine learning classification model based on ECFP4-based Tanimoto similarity between molecules.

(iii) random_forest_regression.py: This script runs random forest regression-based machine learning models based on SEF, Morgan fingerprint and SHED descriptors

(iv) MLP_based_DNN_regression.py and/or MLP_based_DNN_regression_mod.py: These scripts could run MLP-based deep neural network models comparing SEF, Morgan Fingerprint and SHED descriptors. The basic script is 'MLP_based_DNN_regression.py' and more advanced script is 'MLP_based_DNN_regression_mod.py' containing more features of SEF descriptor.

(v) data_prep_SHED.py:This script could build SHED keys (SHED descriptors) from a list of SMILES using 'jCMapperCLI.jar' 


