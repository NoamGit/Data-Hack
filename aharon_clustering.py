
# coding: utf-8

# In[13]:



# In[519]:

from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.cluster import KMeans, MiniBatchKMeans, MeanShift, Birch 
from sklearn.manifold import MDS
from sklearn.mixture import GMM
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
import scipy.cluster.hierarchy as hac
from numpy import linalg as LA
import lda  #https://pypi.python.org/pypi/lda
import os  
import logging
from optparse import OptionParser
import sys
from scipy.stats import chisquare
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import pymongo
from pymongo import MongoClient

import httplib
import numpy as np
import json
import pandas as pd
import math
import time
import urllib
import fastcluster
import os
import re
import email
N_CLUSTERS = 3


# In[ ]:

kw_df = pre_process_df(dff)
tf_idf_data_frame = get_tf_ifd_matrix(kw_df)

model = KMeans(n_clusters=3, random_state=1)
res = model.fit_predict(tf_idf_data_frame)
clusters_centers = model.cluster_centers_


# In[ ]:

path = '/Users/shani/Documents/private/hack/matan/'

data = []
for root, dirs, files in os.walk(path):
    for filename in files:
        with open(path+filename) as data_file:    
            data.append(json.load(data_file))
data = [item for sublist in data for item in sublist]

len([i['Content'] for i in data if "Original Message" in i['Content']])
len(data)


# In[8]:

data[0]


# In[23]:

def addDate(matchobj):
    return '\nDate: ' + matchobj.group(0)

a = []
for l in data:
    c = l['Content']
    if("Original Message" not in c):
        c1 = re.sub('\d\d/\d\d/\d\d\d\d \d\d:\d\d',addDate,c)
        ll = re.split('\nFrom: | \tFrom:',c1)
        if len(ll)>1:
            ok = 1
            for i in ll[1:]:
                e = email.message_from_string("From:" + i)
                if(e['Date'] is None):
                    if(e['When'] is None):
                        ok = 0
            b = e.get_payload()
            re.
            a.append({"date": l['Date'], "content": })
            


# In[24]:

[i['content'] for i in a[:5]]


# In[ ]:


# Phase 2: Remove terms that have too little frequencies - less than 5 
def remove_terms(df, min_num_words_in_total=1):
    df_remove_terms = df.transpose()
    df_remove_terms["count_total_num_words"] = df_remove_terms.sum(1)
    sum_rows = df_remove_terms["count_total_num_words"]
    df_remove_terms = df_remove_terms[df_remove_terms.count_total_num_words > min_num_words_in_total]
    df_remove_terms = df_remove_terms.drop("count_total_num_words",1)
    df_remove_terms = df_remove_terms.transpose()
    print("DF sahpe - after remove words with small frequency:", df_remove_terms.shape)
    return df_remove_terms, sum_rows

# Phase 3: normalize frequencies, by number of words  in documents. 
def normalize_freq(df, sum_rows):
    """ Normalize df - by number of words in document. take a matrix of frequencies and normalize it"""
    # adding the original number of words in document, before removing features (KW)
    df["n_words_in_doc"] = sum_rows  # n_words_in_doc  - used for calculating the freq according
    index_col = len(df.columns) - 1
    df_norm_terms = df.iloc[:,:index_col].div(df["n_words_in_doc"], axis=0)
    return df_norm_terms

def pre_process_df(df):
    print ("df BEFORE :",df.shape
    df_remove_terms,sum_rows = remove_terms(df) 
    final_df = normalize_freq(df_remove_terms,sum_rows) 
    print ("df AFTER:",final_df.shape)
    return final_df
           

def calc_idf(num_docs_for_t,number_docs):  # num_docs_for_t : number of documents where the term  t  appears
     return math.log(1 + number_docs/(1+float(num_docs_for_t)))

def get_idf_for_term(df):
    number_docs = df.shape[0]
    n_terms_doc = []  # get sum of bolean values - in what documenet the term appeared. 
    for term in df.to_dict().values():
        n_terms_doc.append(len([i for i in term.values() if i!=0.]))
    idf_for_term = []
    for term in n_terms_doc:
        idf_for_term.append(calc_idf(term,number_docs))
    return idf_for_term

def get_tf_ifd_matrix(df):
    idf_for_term = get_idf_for_term(df)
    d = df.transpose()  
    tf_idf_list = []
    for i in range(d.shape[0]): # as nubmber of words
        tf_idf_list.append([idf_for_term[i]*j for j in d.ix[i]])
    # Matrix: terms in columns, docs as index. 
    tf_idf_data_frame = pd.DataFrame(tf_idf_list,columns=d.columns).transpose()
    #print("head", tf_idf_data_frame.head())
    #print(tf_idf_data_frame.shape)
    tf_idf_data_frame.columns = df.columns
    return tf_idf_data_frame


# In[ ]:

def hieraric_clustering(df,th=1.):
    X = np.array(df)
    dist_matrix = pdist(X, 'euclidean')
    #t = fastcluster.linkage(dist_matrix, method='single', metric='euclidean', preserve_input='True')
    hc = linkage(dist_matrix)
    ttt = fcluster(hc,th)
    #print (Counter(ttt))
    return ttt


def determine_dis():
    distnaces = np.arange(0.8,1.5,0.0001)
    x_vals = []
    y_vals = []
    for i in distnaces:
        ttt = fcluster(hc,i)
        y_vals.append(len(list(set(ttt))))
        x_vals.append(i)
    plt.plot(x_vals,y_vals)
    plt.ylabel('number_of_clusters')
    plt.show()
    


#  topics_dict = filter_topics_in(all_data)
#  topics_df = build_df(topics_dict)
    
def get_topic_classes(all_data, dff=None):
    """ Returns sub data of the original problem - for one topic cluster result"""
    if dff is None:
        dff = data_after_process(all_data)
    # cluster by topics, after cleaning. 
    # get the KW data-frame:
    kw_df = pre_process_df(dff)
    print(kw_df.shape)
    # Get the topics data-frame
    topics_dict = filter_topics_in(all_data)
    topics_df = build_df(topics_dict)
    topics_df = topics_df.drop("n_words_in_doc",1)
    print ("topic df before removing bad docs:", topics_df.shape)
    # remove documents that were removed in the pro-process stage: 
    topics_df = topics_df[topics_df.index.isin(np.array(kw_df.index))]
    print ("topic df after removing bad docs:", topics_df.shape)
    cluster_docs_res = get_one_topic_res(topics_df)
    print ("cluster_docs_res head", cluster_docs_res.head())
    # add topic clustering results to the keywords dataframe
    res_label = pd.concat([cluster_docs_res, dff], axis=1)
    #one_topic_label = res_label[res_label["doc_topic_class"] == 0]
    #second_topic_label = res_label[res_label["doc_topic_class"] == 1]
    #print ("one_topic_data frame:",one_topic_label.shape)
    #return one_topic_label,second_topic_label
    return res_label



# In[454]:

# filter the topics out:
def filter_out_topics(all_data,filtered_qw,qw_id_names_dict):
    all_data_kw_only = {}
    for doc in all_data.items():
        one_doc_kw_dict = {}
        for kw_list in doc[1]["keywords"]:
            kw_id = kw_list[u'id']
            if kw_id in filtered_qw:
                one_doc_kw_dict[qw_id_names_dict[kw_id]] = kw_list[u'c']
        all_data_kw_only[doc[0]] = one_doc_kw_dict
    return all_data_kw_only

def filter_kw_by_kl(all_kw):
    kw_mongo = kwd_kl_collec.find({"value.kl": {"$gt": 1}, "_id": {"$in":all_kw}},[])
    filtered_qw = []
    for i in kw_mongo:
        filtered_qw.append(int(i['_id']))
    return filtered_qw


def build_df(all_data_filter,df_type = "kw"):
    df = pd.DataFrame(all_data_filter.values())
    df.index = all_data_filter.keys()
    df = df.fillna(0)
    df.columns = [str(i) for i in df.columns]
    if df_type == 'topics':
        sum_name = "n_topics_in_doc"
    else:
        sum_name = "n_words_in_doc"
    df[sum_name] = df.sum(1)
    return df

def get_id_names_dict(all_kw):
    kw_names_mongo = keywords_colle.find({"_id": {"$in":all_kw}},["_id","l"])
    kw_names_mongo_data = {}
    for i in kw_names_mongo:
        kw_names_mongo_data[i['_id']]= i["l"]
    return kw_names_mongo_data


def data_after_process(all_data):
    all_kw = list(set([item for sublist in [[i["id"] for i in one_doc [1]['keywords']] for one_doc in all_data.items()] for item in sublist]))
    print ("len of all key words:",len(all_kw))
    filtered_qw = filter_kw_by_kl(all_kw)
    qw_id_names_dict = get_id_names_dict(all_kw)
    filtered_docs = filter_out_topics(all_data,filtered_qw,qw_id_names_dict)
    df = build_df(filtered_docs)
    return df

def file_to_df(file_name):
    f = open(file_name, 'r')
    in_js_format = f.read()
    dict_of_docs = json.loads(in_js_format)
    df = pd.DataFrame(dict_of_docs.values())
    df.index = dict_of_docs.keys()
    df = df.fillna(0)
    df["n_words_in_doc"] = df.sum(1)
    #print ("data frame shpae - strting", df.shape)
    return df




# In[ ]:

# Creating  tf-idf matrix:
# In[517]:

def calc_bolean(col):
    return np.sum([1 for i in col if i!=0.])



# Feature Reduction
# In[8]:

def feature_red_las(df,n_components=250):
    svd = TruncatedSVD(n_components=n_components, random_state=1)
    svd_kw_df = svd.fit_transform(df)
    svd_kw_df = pd.DataFrame(svd_kw_df)
    #print ("Data frame, after dimention reduction in LSA: ",svd_kw_df.shape)
    return svd_kw_df
    
#print ("tf_idf_data_frame - shouldn't chnage , after dimention reduction in LSA: ",tf_idf_data_frame.shape)


# In[498]:

def calc_distance(vec1,vec2):
    "Calculate distance between vectors"
    norm = LA.norm(vec1)*LA.norm(vec2)
    return sum(vec1*vec2)/norm


def build_model(df, cluster_type="kmeans", seed=1):
    if cluster_type == "birch":
        model = Birch(n_clusters=N_CLUSTERS)
        res = model.fit_predict(df)
    elif cluster_type == "minibatch":
        model = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed)
        res = model.fit_predict(df)
    elif cluster_type == "em":
        model = mixture.GMM(n_components=N_CLUSTERS)
        model.fit(df)
        res = model.predict(df)
    elif cluster_type == 'lda':
        model = lda.LDA(n_topics=N_CLUSTERS, n_iter=1500, random_state=seed)
        data_to_cluster = np.array(df).astype(int)
        lda_res = model.fit_transform(data_to_cluster)
        res = []
        for i in lda_res:  #for now - do hard clustering, take the higheset propability
            res.append(i.argmax())
    else:
        model = KMeans(n_clusters=N_CLUSTERS, random_state=seed)
        res = model.fit_predict(df)
        df_array = np.array(df)

        dis_dict = {}
        for i in range(N_CLUSTERS):
            dis_dict[i] = clusters_centers[i]
        all_dist = []
        for line_idx in range(len(df_array)):
            label =  model.labels_[line_idx]
            dist = calc_distance(df_array[line_idx],dis_dict[label])
            all_dist.append(dist)
        df["distance_from_cluster"] = all_dist

    #clusters = model.labels_.tolist()
    #print ("clusters are:",clusters)
    print(""">>>> model is: %s, # of clusters:%s, and %s""" %(cluster_type,N_CLUSTERS,Counter(res)))
    res = [str(i) for i in res]
    docs_clusteres = zip(df.index,res)
    return docs_clusteres


def cluster_and_plot(df_to_cluster,cluster_type,original_df=None):
    if original_df is None:
        original_df = df_to_cluster
    print ("df to cluster - shape %s and original_df shape: %s" % (df_to_cluster.shape, original_df.shape))
    res = build_model(df_to_cluster, cluster_type)
    #clusters = [i[1] for i in res]
    #plot_clustering_res(original_df, clusters)
    return res


# Run Cluster: 
def cluster_docs(df, pre_process="tf-idf",cluster_type = "kmeans"):
    #final_df = pre_process_df(df) - No need in case we do topic clustering first..
    final_df = df
    if pre_process == "tf-idf":
        # Get tf-idf matrix: 
        start_process_time = time.time()
        tf_idf_data_frame = get_tf_ifd_matrix(final_df)
        finish_process_time = time.time()
        print ("process data - time it took %s" % (finish_process_time - start_process_time))
        res = cluster_and_plot(tf_idf_data_frame,cluster_type,final_df)
    elif pre_process == "lsa":
        #print ("removing featrues in LSA:")
        #final_df = pre_process_df(df)
        df_lsa = feature_red_las(final_df)
        res = cluster_and_plot(df_lsa,cluster_type,final_df)
    elif pre_process == "clean": 
        #final_df = pre_process_df(df)
        res = cluster_and_plot(final_df,cluster_type)
    elif pre_process == "norm":
        df1 = df.drop("n_words_in_doc",1)
        norm = StandardScaler()
        df2 = norm.fit_transform(df1)
        df2 = pd.DataFrame(df2, columns = df1.columns, index = df1.index)
        res = cluster_and_plot(df2,cluster_type)
    elif pre_process == "norm_freq":
        df4 = normalize_freq(df) # normalize frew removes the last column of the count words..
        df4 = df4 * 100.
        res = cluster_and_plot(df4,cluster_type)
    else:  # Don't do any process on data, except replace NaN with zero
        df1 = df.drop("n_words_in_doc",1)
        res = cluster_and_plot(df1,cluster_type)
    return res



# In[ ]:




# In[64]:

def write_res_to_file(search_w, method_list ,all_res): #method - nothing_em for example
    for method in method_list:
        res_df = pd.DataFrame(all_res[method],columns = ["id","label"])
        path_name = "/Users/shani/git_code/%s_list_of_docs_ids_%s.csv"% (search_w, method)
        res_df.to_csv(path_name)

def get_results_for_sw(data_frame,sw):
    #print("File name: %s " % filename)
    all_res = {}
    process_type_list = ["tf-idf","lsa","clean","nothing","norm","norm_freq"]
    cluster_type_list = ["kmeans","birch","minibatch","em"]
    
    #df = file_to_df(filename)
    for process_type in process_type_list:
        for cluster_type in cluster_type_list:
            print ("pre-process is %s , cluster type is: %s" % (process_type, cluster_type))
            c_p = process_type + "_" + cluster_type
            start = time.time()
            all_res[c_p] = cluster_docs(data_frame, pre_process=process_type, cluster_type=cluster_type)
            print ("time that took: %s" % (time.time() - start))
    #write_res_to_file(sw,["nothing_em","tf-idf_em","nothing_lda","norm_freq_lda","norm_lda"],all_res)
    return all_res


# In[ ]:




# In[ ]:




# In[12]:

get_ipython().magic(u'matplotlib inline')
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5:'#eeefff'}

def plot_clustering_res(df_to_plot,clusters_of_model):
    #print ("DF to plot - shape:", df_to_plot.shape)
    #print ("clusters_of_model len - ", len(clusters_of_model))
    MDS()
    dist = 1 - cosine_similarity(df_to_plot)  # after SVD
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    #create data frame that has the result of the MDS plus the cluster numbers and titles
    #group by cluster
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters_of_model)) 
    groups = df.groupby('label')
    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, color=cluster_colors[name], 
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    ax.legend(numpoints=1)  #show legend with only 1 point
    plt.show() #show the plot

    #plt.savefig('clusters_small_noaxes.png', dpi=200)


# LDA 

# In[ ]:




# In[330]:

start = time.time()
df1 = dff.drop("n_words_in_doc",1)
#df1 = normalize_freq(df) # normalize frew removes the last column of the count words..
#df1 = df1 * 100
lda_model = lda.LDA(n_topics=4, n_iter=1500, random_state=1)
data_to_cluster = np.array(df1).astype(int)
lda_res = lda_model.fit_transform(data_to_cluster)



# In[331]:

lda_res_hard_cluster = []
for i in lda_res:  #for now - do hard clustering, take the higheset propability
    lda_res_hard_cluster.append(i.argmax())
print (Counter(lda_res_hard_cluster)) 
end = time.time() 



# In[332]:

topic_word = lda_model.topic_word_ 
n_top_words = 15
vocab = np.array(df1.columns)
for i, topic_dist in enumerate(topic_word):
    #topic_words = [np.array(df1.columns)(ks) for ks in np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]]
    topic_words =  np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ','.join(topic_words)))
    

    


# In[ ]:




# In[ ]:




# In[ ]:




# In[180]:

topic_word = lda_model.topic_word_ 
n_top_words = 20
vocab = np.array(df1.columns)
for i, topic_dist in enumerate(topic_word):
    #topic_words = [id_kw_dict.get(ks,"not-found?") for ks in np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]]
    topic_words =  np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ','.join(topic_words)))

#lda_model.topic_word_
#lda_model.components_[0]
#vocab[np.argsort(topic_word[0])]
#lda_model.doc_topic_


# In[49]:

#plot_clustering_res(df1, lda_res)
lda = zip(df1.index, lda_res_hard_cluster)
#lda_res_hard_cluster


# In[52]:

#write_res_to_file("lda-fitbit-res.txt", lda)


# In[148]:

model = mixture.GMM(n_components=N_CLUSTERS)
model.fit(df1)
res = model.predict(df1)


# In[ ]:




# In[114]:

#df_1 = data_after_process(all_data)


# In[118]:

#df_1["doc_id"] = df_1.index
#df_1.index = range(df_1.shape[0])
df_1.head()
dff = df_1.drop('n_words_in_doc',1)


# In[ ]:

a = pd.DataFrame([[1,2,3,5,7],[4,5,6,0,0],[7,8,9,1,1]])
a.columns = ["a","b","c","f","g"]
a.index = ["g","k","j"]
a


# In[601]:

a.loc[:,["a","g"]]


# In[344]:



a_trams = np.array(a.transpose())
a = np.array(a)
a_trams


# In[347]:

#np.array(a_trams)*np.array(a)
np.mat(a_trams) * np.mat(a) 


# In[597]:

b = pd.DataFrame([[2,33],[5,66]])
b.columns = ["b","class"]
b.index = ["g","k"]
#pd.merge(a,b,axis=0)
g = pd.concat([a, b], axis=1,join_axes=[b.index])


# Write clustering results to Json

# In[596]:

b


# In[13]:

def write_res_to_file(output_file, res):
    res_to_json = []
    for i in res:
        res_to_json.append({i[0]:str(i[1])})
    json_res = json.dumps(res_to_json) 
    with open(output_file, 'w') as outfile:
        json.dump(res_to_json, outfile)
    


# In[296]:

a
a.loc[10] = np.array([2, 3, 4])
a


# Chi2:
#     1. Do chi2 for dimension reduction (will be done on all sapce)
#     2. Choose best words - Do chi2 on each cluster / combine it with lift calc.  
# 

# In[136]:

# Read the dict of id-kw
f = open('Fitbit-kw-id-name-dict.txt', 'r')
kw_id_in_js_format = f.read()
list_of_kw_ids = json.loads(kw_id_in_js_format)
list_of_kw_ids
id_kw_dict = {}
for i in list_of_kw_ids:
    id_kw_dict[unicode(i['_id'])] = i['l']
    #print (i['_id'],i['l'])   


# In[ ]:




# In[50]:

# Do chi2 for feature selection: 

chi2,p=chisquare(np.array(tf_idf_data_frame)) 
#tf_idf_data_frame.head()  - lost the keywords on the way.
kw_res = zip(tf_idf_data_frame.columns,chi2)
kw_res.sort(key = lambda t: t[1],reverse=True)
print ([i[0] for i in kw_res[:50]])


# In[84]:

[id_kw_dict.get(i[0],-1) for i in kw_res[:50]]


# In[166]:

a = {"a":1,"b":1,"a":2}


# In[167]:

a


# In[ ]:




# In[ ]:




# In[14]:

# To get data from saved file.. 
def get_data_from_saved_file():
    f = open('apple_ky_adi_res_json.txt', 'r')
    kw_id_in_js_format = f.read()
    all_data = json.loads(kw_id_in_js_format)
    dff = pd.DataFrame(all_data.values(),index=all_data.keys())
    dff = dff.fillna(0)
    dff["n_words_in_doc"] = dff.sum(1)
    dff.shape
    return dff

#dff = get_data_from_saved_file()


# In[ ]:

# Plot the number of docs for word...

#plt.plot(np.sort(np.array(dff["n_words_in_doc"])))
#plt.ylabel('length of docs')
#plt.show()

#np.sort(np.array(dff["n_words_in_doc"]))[100:]
#len(np.array(dff["n_words_in_doc"]))


# Extract good KW by Logistic regression:
def get_kw_features():
    res = cluster_docs(dff, pre_process="tf-idf", cluster_type="em")
    r = dict(res)
    res_df = pd.DataFrame(r.values())
    res_df.columns = ["class"]
    res_df.index = r.keys()
    print("Finished clustering")
    #concatenate the DF and the clustering results
    res_label = pd.concat([res_df, dff], axis=1)
    all_res_label=res_label.drop("n_words_in_doc",1)
    all_res_label.shape

    # SUM all results, and run algorithm (LR) to identify important words
    all_g = all_res_label.groupby("class").sum()
    x = np.array(all_g)
    y = np.array(all_g.index)
    lr_model = LogisticRegression(penalty="l1",C=10)
    lr_model.fit(x, y)
    
chi2,p=chisquare(np.array(all_g)) 
rr = np.array(all_g.columns)[np.argsort(chi2)[::-1]]
#for i in range(40):
 #   print (rr[i])
    

