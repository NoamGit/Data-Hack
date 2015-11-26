
# coding: utf-8

# In[121]:

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression 
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import math
from numpy import linalg as LA
from sklearn.ensemble import RandomForestClassifier as RF
import random
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler
SEED = 1000


# In[185]:

#d = pd.read_csv("jona_tags_with_parser.csv")
#k  = pd.read_csv("all_kw_with_index_rank.csv")

org = pd.read_csv("index_urls_for_pd.csv")
org = org.dropna(axis=0, how='all')

new_tags = pd.read_csv("index_url_for_pd_jona_tags.csv")
#new_tags = new_tags[new_tags.isDefinitelyBad != "na"]
#new_tags = StandardScaler().fit_transform(new_tags)
all_f = [u'content-size', u'abs-ratio','number-stop-words-in-html',  u'score2', u'score3',u'stop-words-raw-content',
       u'number-of-sentences-with-stop', u'number-scores', u'ratio13', u'score1',
       u'ratio23',  u'len-data', u'ratio',
       u'real-ratio12']

not_now = [u'is-index-new', u'neg_ind', u'pos_ind',
       u' suspected-index-page-lr-old', u'suspected-index-page.1',
       u' suspected-junk-page',u'suspected-index-page', u'n-sen',
       u'vol', u'index_square', u'content-size', u' score2', u' score3',u'stop-words-raw-content',
       u'number-of-sentences-with-stop', u'number-stop-words-in-html',
       u'number-scores', u' ratio13', u' score1',
       u'ratio23', u'abs-ratio', u'len-data', u'ratio',
       u'real-ratio12']


# In[188]:

t = run_many_times()
t


# In[189]:

t


# In[181]:

## Run model on not index pages...
def run_on_all_space():
    model = LogisticRegression(penalty="l2",C=0.08)
    new_tags = pd.read_csv("index_url_for_pd_jona_tags.csv")
    new_tags = new_tags.reindex(np.random.permutation(new_tags.index))
    model.fit(new_tags[all_f],new_tags[st])
    t = np.append(model.coef_, model.intercept_)
    return t

def run_many_times():
    all_coeff = []
    names = []
    for i in range(10):
        new_tags = pd.read_csv("index_url_for_pd_jona_tags.csv")
        new_tags = new_tags.reindex(np.random.permutation(new_tags.index))
        a = run(new_tags, "isDefinitelyBad", 11) 
        all_coeff.append(a[0])
        names.append(a[1])
    return all_coeff, names

def run(df, st, seed):
    train_df, test_df = train_test(df, seed)
    res = run_model(all_f, train_df, test_df, st, c=0.08)
    feature_import(train_df,test_df, st,all_f)
    run_c(train_df, test_df,st)
    return res

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def process_data(d,k):
    r = k['index']
    k['norm_index'] = [i/LA.norm(r) for i in r]
    #kk = k[["org-url","n-sen","vol","index_square",u'index','norm_index']]
    d['org-url'] = d['url']
    df = pd.merge(d, k, right_index=True, how="inner",on="org-url")
    df = df.fillna(0)
    return df

def train_test(df,seed):
    prng = RandomState(seed)
    train_index = np.random.permutation(int(0.8*df.shape[0]))
    test_index = [i for i in range(df.shape[0]) if i not in train_index]
    #print "train intrx", train_index
    train_df = df.iloc[train_index,:]
    test_df = df.iloc[test_index,:]
    print "train df shape:", train_df.shape ," test df shape:", test_df.shape
    return train_df, test_df


def run_model(all_f, train_df, test_df, st, c=0.08, csv_name=None):
    model = LogisticRegression(penalty="l2",C=c)
    tr = train_df[all_f]
    tst = test_df[all_f]
    #norm = StandardScaler()
    #norm.fit(tr)
    #tr = norm.transform(tr)
    #tst = norm.transform(tst)
    model.fit(tr,train_df[st])
    y_pred = model.predict(tst)
    c =  confusion_matrix(test_df[st], y_pred)
    print ">>>>> "
    print c
    print calc_recall(c)
    #print sorted(zip(model.coef_[0], train_df[all_f].columns),key=lambda x: x[0],reverse=True)
    #print zip(model.coef_[0], train_df[all_f].columns)
    if csv_name:
        test_df["pred"] = y_pred
        test_df.to_csv(csv_name)
    return np.append(model.coef_, model.intercept_),tr[all_f].columns #,y_pred,model.predict_proba

def feature_import(train_df,test_df, ds,all_f):
    m = RF()
    m.fit(train_df[all_f],train_df[ds])
    y = m.predict(test_df[all_f])
    z = zip(m.feature_importances_,train_df[all_f].columns)
    c =confusion_matrix(y,test_df[ds])
    print "RF res c:",c
    print sorted(z, key=lambda x: x[0],reverse=True)

def calc_recall(c):
    tp = float(c[0][0])
    fp = float(c[0][1])
    fn = float(c[1][0])
    tn = float(c[1][1])
    try:
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)
        sps = float(tn) / (tn + fp)
        accuracy = float(tp + tn) / (tp + tn + fp + fn)
        print "precision:", precision, "recall:", recall, "sps:", sps, "accuracy:", accuracy
    except:
        print "there was a prblem - division in zero"
    #return precision, recall, sps, accuracy

def predict_index(coeff,test_df,all_f):
    b = []
    for i in (range (test_df.shape[0])):
        v = np.append(np.array(test_df.loc[:,all_f])[i],[1])
        s = sigmoid(np.inner(coeff,v))
        b.append(0 if (s < 0.5) else 1)
    return b

def run_c(train_df, test_df,st):
    for c in [0.001,0.02,0.04,0.06, 0.08,0.1,0.2,0.3,0.4,0.8,1]:
        model = LogisticRegression(penalty="l2",C=c)
        model.fit(train_df[all_f],train_df[st])
        y_pred = model.predict(test_df[all_f])
        y_train_pred =  model.predict(train_df[all_f])
        print "c is :", c, "and accuracy score:" , accuracy_score(test_df[st], y_pred)
        print "test confuaion m:"
        print confusion_matrix(test_df[st], y_pred)
        #print "and train cf:"
        #print confusion_matrix(train_df[st], y_train_pred)


# In[4]:

## Run classifier on raw data. 


# In[5]:




# In[9]:

a = {}
b = []
c = np.append(model.coef_[0],model.intercept_)
for i in (range (test_df.shape[0])):
    v = np.append(np.array(test_df.loc[:,all_f])[i],[1])
    s = sigmoid(np.inner(c,v))
    a[np.array(test_df.loc[:,"url"])[i]] = 1 if (s < 0.5) else 0
    b.append(0 if (s < 0.5) else 1)
    
    
### Try to  normalize..
tmp_tr_df = train_df[all_f]
tmp_test_df = test_df[all_f]
norm_tr_df = (tmp_tr_df - tmp_tr_df.mean()) / (tmp_tr_df.max() - tmp_tr_df.min())
norm_test_df = (tmp_test_df - tmp_test_df.mean()) / (tmp_test_df.max() - tmp_test_df.min())

model = LogisticRegression(penalty="l2",C=0.04)
model.fit(norm_tr_df,y_train)
y_pred = model.predict(norm_test_df)

print confusion_matrix(y_test, y_pred)
print accuracy_score(y_test, y_pred)
test_df["pres"] = y_pred
test_df["real"] = y_test

#test_df.to_csv("jona_test_res.csv")



#run(new_tags, "is_index_real", 10) 
#
train_df, test_df = train_test(new_tags, 10)
st = "isDefinitelyBad"
tr = train_df[all_f]
tst = test_df[all_f]
norm = StandardScaler()
norm.fit(tr)
tr = norm.transform(tr)
tst = norm.transform(tst)
model.fit(tr,train_df[st])
y_pred = model.predict(tst)
c =  confusion_matrix(test_df[st], y_pred)

print "c is: ", c
y_pred_prob =  model.predict_proba(test_df[all_f])
#print y_pred#
#print test_df["isDefinitelyBad"]
#train_df[all_f]

#r = run(new_tags, "isDefinitelyBad", 10) 
#r[2]
#train_df[all_f]
zip(model.coef_[0], train_df[all_f].columns)
print test_df[st]
print [i[0] for i in y_pred_prob]


