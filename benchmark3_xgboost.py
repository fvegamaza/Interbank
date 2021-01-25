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

train.to_csv("train_bench.csv")
test.to_csv("test_bench.csv")

#Tuning hyperparametres Here


folds = [train.index[t] for t, v in KFold(5).split(train)]


from sklearn.model_selection import ParameterGrid

params = ParameterGrid({"min_child_samples": [150, 250, 500, 1000], "boosting_type": ["gbdt", "goss"]})

best_score = 0
best_probs = []
for param in params:
    test_probs = []
    train_probs = []
    p  = "///".join([f"{k}={v}" for k, v in param.items()])
    print("*"*10, p, "*"*10)
    for i, idx in enumerate(folds):
        Xt = train.loc[idx]
        yt = y_train.loc[Xt.index]

        Xv = train.drop(Xt.index)
        yv = y_train.loc[Xv.index]

        learner = LGBMClassifier(n_estimators=1000, **param)
        learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="auc",
                    eval_set=[(Xt, yt), (Xv, yv)], verbose=False)
        test_probs.append(pd.Series(learner.predict_proba(test)[:, -1], index=test.index, name="fold_" + str(i)))
        train_probs.append(pd.Series(learner.predict_proba(Xv)[:, -1], index=Xv.index, name="probs"))

    test_probs = pd.concat(test_probs, axis=1).mean(axis=1)
    train_probs = pd.concat(train_probs)
    score = roc_auc_score(y_train, train_probs.loc[y_train.index])
    print(f"roc auc estimado para {p}: {score}")
    if score > best_score:
        print("*"*10, f"{p} es el nuevo mejor modelo", "*"*10)
        best_score = score
        best_probs = test_probs

best_probs.name = "target"
best_probs.to_csv("benchmark3_xgboost.csv")