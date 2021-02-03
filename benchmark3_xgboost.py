import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import re
import os
path = os.getcwd()
rcc_train = pd.read_csv(path + "\\kaggle\\rcc_train.csv")
se_train = pd.read_csv(path + "\\kaggle\\se_train.csv", index_col="key_value")
sunat_train = pd.read_csv(path + "\\kaggle\\sunat_train.csv")
y_train = pd.read_csv(path + "\\kaggle\\y_train.csv", index_col="key_value").target

rcc_test= pd.read_csv(path + "\\kaggle\\rcc_test.csv")
se_test= pd.read_csv(path + "\\kaggle\\se_test.csv", index_col="key_value")
sunat_test= pd.read_csv(path + "\\kaggle\\sunat_test.csv")


rcc_train[(rcc_train.key_value == 4) & (rcc_train.cod_instit_financiera == 33)].sort_values("codmes")
rcc_train[(rcc_train.key_value == 4) & (rcc_train.cod_instit_financiera == 61)].sort_values("codmes")


bins = [-1, 0, 10, 20, 30, 60, 90, 180, 360, 720, float("inf")]
rcc_train["condicion"] = pd.cut(rcc_train.condicion, bins)
rcc_train["condicion"] = rcc_train["condicion"].cat.codes
rcc_test["condicion"] = pd.cut(rcc_test.condicion, bins)
rcc_test["condicion"] = rcc_test["condicion"].cat.codes

def makeCt(df, c, aggfunc=sum):
    try:
        ct = pd.crosstab(df.key_value, df[c].fillna("N/A"), values=df.saldo, aggfunc=aggfunc)
    except:
        ct = pd.crosstab(df.key_value, df[c], values=df.saldo, aggfunc=aggfunc)
    ct.columns = [f"{c}_{aggfunc.__name__}_{v}" for v in ct.columns]
    return ct

train = []
test = []
aggfuncs = [len, sum, min, max]
for c in rcc_train.drop(["codmes", "key_value", "saldo"], axis=1):
    print("haciendo", c)
    train.extend([makeCt(rcc_train, c, aggfunc) for aggfunc in aggfuncs])
    test.extend([makeCt(rcc_test, c, aggfunc) for aggfunc in aggfuncs])

pd.crosstab(rcc_train.key_value, rcc_train.condicion.fillna("N/A"), values=rcc_train.saldo, aggfunc="mean")

import gc
del rcc_train, rcc_test
gc.collect()

train = pd.concat(train, axis=1)
test = pd.concat(test, axis=1)

pd.crosstab(sunat_train.key_value, sunat_train.ciiu)

train = train.join(pd.crosstab(sunat_train.key_value, sunat_train.ciiu)).join(se_train)
test = test.join(pd.crosstab(sunat_test.key_value, sunat_test.ciiu)).join(se_test)
del sunat_train, se_train, sunat_test, se_test
gc.collect()
#keep the same columns in both datasets
keep_cols = list(set(train.columns).intersection(set(test.columns)))
train = train[keep_cols]
test = test[keep_cols]
len(set(train.columns) - set(test.columns)) , len(set(test.columns) - set(train.columns))

train.columns = [str(c) for c in train.columns]
train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_-]+', '', x))

test.columns = [str(c) for c in test.columns]
test = test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_-]+', '', x))

#Read dataseeeeeeeeeeeeeeeets

train = pd.read_csv(path + "\\train_bench.csv",index_col="key_value")
test = pd.read_csv(path + "\\test_bench.csv",index_col="key_value")
y_train = pd.read_csv(path + "\\kaggle\\y_train.csv", index_col="key_value").target


from sklearn.model_selection import RandomizedSearchCV
model = XGBClassifier()
model.fit(train,y_train)
predict_train = model.predict_proba(train)

accuracy_train = accuracy_score(y_train,predict_train)
print('\naccuracy_score on train dataset : ', accuracy_train)

predict_test = model.predict_proba(test)
accuracy_score(test,predict_test)
predict_test_result = pd.DataFrame()
predict_test_result["target"]
predict_test[:,0]
prediction = []


predict_test_result["target"] = predict_test[:,1]
predict_test_result.to_csv("benchmark3_xgboost.csv")
print(predict_test_result.index.name)# = "key_value"
np.count_nonzero(y_train == 1)

# A parameter grid for XGBoost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

estimator = XGBClassifier(
    objective= 'reg:squarederror',
    nthread=4,
    seed=42
)

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}

grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 10,
    cv = 10,
    verbose=True
)

grid_search.fit(train, y_train)

grid_search.best_estimator_