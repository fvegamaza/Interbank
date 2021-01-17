import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import re
import os
path = os.getcwd()


rcc_train = pd.read_csv(path + "\\kaggle\\rcc_train.csv")
se_train = pd.read_csv(path + "\\kaggle\\se_train.csv", index_col="key_value")
censo_train = pd.read_csv(path + "\\kaggle\\censo_train.csv", index_col="key_value")
y_train = pd.read_csv(path + "\\kaggle\\y_train.csv", index_col="key_value").target

rcc_test= pd.read_csv(path + "\\kaggle\\rcc_test.csv")
se_test= pd.read_csv(path + "\\kaggle\\se_test.csv", index_col="key_value")
censo_test= pd.read_csv(path + "\\kaggle\\censo_test.csv", index_col="key_value")


bins = [-1, 0, 10, 20, 30, 60, 90, 180, 360, 720, float("inf")]
rcc_train["condicion"] = pd.cut(rcc_train.condicion, bins)
rcc_test["condicion"] = pd.cut(rcc_test.condicion, bins)

def makeCt(df, c, aggfunc=sum):
    try:
        ct = pd.crosstab(df.key_value, df[c].fillna("N/A"), values=df.saldo, aggfunc=aggfunc)
    except:
        ct = pd.crosstab(df.key_value, df[c], values=df.saldo, aggfunc=aggfunc)
    ct.columns = [f"{c}_{aggfunc.__name__}_{v}" for v in ct.columns]
    return ct


train = []
test = []
aggfuncs = [len, sum]
for c in rcc_train.drop(["codmes", "key_value", "saldo"], axis=1):
    print("haciendo", c)
    train.extend([makeCt(rcc_train, c, aggfunc) for aggfunc in aggfuncs])
    test.extend([makeCt(rcc_test, c, aggfunc) for aggfunc in aggfuncs])

train = pd.concat(train, axis=1)
test = pd.concat(test, axis=1)

train = train.join(censo_train).join(se_train)
test = test.join(censo_test).join(se_test)

keep_cols = list(set(train.columns).intersection(set(test.columns)))
train = train[keep_cols]
test = test[keep_cols]
len(set(train.columns) - set(test.columns)) , len(set(test.columns) - set(train.columns))

test = test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_-]+', '', x))
train = train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_-]+', '', x))

folds = [train.index[t] for t, v in KFold(5).split(train)]
test_probs = []
train_probs = []
fi = []
for i, idx in enumerate(folds):
    print("*" * 10, i, "*" * 10)
    Xt = train.loc[idx]
    yt = y_train.loc[Xt.index]

    Xv = train.drop(Xt.index)
    yv = y_train.loc[Xv.index]

    learner = LGBMClassifier(n_estimators=1000)
    learner.fit(Xt, yt, early_stopping_rounds=10, eval_metric="auc",
                eval_set=[(Xt, yt), (Xv, yv)], verbose=50)
    test_probs.append(pd.Series(learner.predict_proba(test)[:, -1], index=test.index, name="fold_" + str(i)))
    train_probs.append(pd.Series(learner.predict_proba(Xv)[:, -1], index=Xv.index, name="probs"))
    fi.append(pd.Series(learner.feature_importances_ / learner.feature_importances_.sum(), index=Xt.columns))

test_probs = pd.concat(test_probs, axis=1).mean(axis=1)
train_probs = pd.concat(train_probs)
fi = pd.concat(fi, axis=1).mean(axis=1)

print("roc auc estimado: ", roc_auc_score(y_train, train_probs.loc[y_train.index]))

test_probs.name = "target"
test_probs.to_csv("benchmark1.csv")

