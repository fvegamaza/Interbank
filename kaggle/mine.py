import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import re
import os
path = os.getcwd()
#Before tuning hypeparameters
train = pd.read_csv(path + "\\train_bench.csv")
test = pd.read_csv(path + "\\test_bench.csv")
y_train = pd.read_csv(path + "\\kaggle\\y_train.csv", index_col="key_value").target
del train["key_value"], test["key_value"]

feature_imp = pd.read_csv(path + "\\feature_imp.csv")
feature_imp_list = feature_imp["Feature"].values.to_list()

train[train.name.isin(feature_imp_list)]