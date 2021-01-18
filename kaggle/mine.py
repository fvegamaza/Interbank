import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import re
import os
import matplotlib.pyplot as plt
path = os.getcwd()
#Before tuning hypeparameters
train = pd.read_csv(path + "\\train_bench.csv",index_col="key_value")
test = pd.read_csv(path + "\\test_bench.csv",index_col="key_value")
y_train = pd.read_csv(path + "\\kaggle\\y_train.csv", index_col="key_value").target

#feature_imp = pd.read_csv(path + "\\feature_imp.csv")
#feature_imp_list = feature_imp["Feature"].to_list()
#feature_imp_list = feature_imp_list[200:] #best features, i think
#train_select = train[feature_imp_list]

#Check if the data is logical
plt.hist(train["dto"])#, range= (0,0.43697))
train["RIESGO_DIRECTO_max_1"]

#Cod_ubi ############################
train["cod_ubi"].value_counts().head(15)
train.cod_ubi.mean() # Es cero
train.cod_ubi.std()
cod_ubi_unique = train["cod_ubi"].unique()
cod_ubi_unique_sorted = np.sort(cod_ubi_unique)

np.set_printoptions(edgeitems=30)
np.diff(cod_ubi_unique_sorted)#most of the values are 1.13659769e-05
np.diff(cod_ubi_unique_sorted/0.000011365976900000001)

(train["cod_ubi"]/0.000011365976900000001).head(20) # i see .550  in positive int and .450 in neg
(train["cod_ubi"]/0.000011365976900000001 - .550281).head(20)
new_cod_ubi = train["cod_ubi"]/0.000011365976900000001
new_var = []
for i in new_cod_ubi:
    if i >= 0 :
        result = (i - .550281)
    else:
        result = (i + .449992)
    new_var.append(result)
new_var = pd.DataFrame(new_var,columns=["cod_ubi_reverse"])

train["cod_ubi"].isnull().sum()
new_var.isnull().sum()

#Edad ######
train["edad"].value_counts().head(30)
train.edad.mean()
train.edad.std()
edad_unique = train["edad"].unique()
edad_unique_sorted = np.sort(edad_unique)
np.diff(edad_unique_sorted)# Seems repeated values
np.diff(edad_unique_sorted/0.07573877) #almost 1

(train["edad"]/0.07573877 - .012577).head(25)

new_cod_ubi = train["edad"]/0.07573877
new_var = []
for i in new_cod_ubi:
    if i >= 0 :
        result = (i - .012577)
    else:
        result = (i + .987421)
    new_var.append(result)
new_var = pd.DataFrame(new_var,columns=["edad_reverse"])
new_var["edad_reverse"].value_counts().head(20).sum()