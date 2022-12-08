from time import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy import zeros
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import normalization, Proprecess, getGraph, preprocess_adj, performance

max_iter = 50
dim = 200

start = time()
# read data
data = pd.read_csv('data/Biase.csv',header=None)
data = Proprecess(data)
data = normalization(data)
features = data

# read label
y = pd.read_csv('data/Biase_true_labs.csv',header=None,low_memory=False)
y = np.array(y)
y = y.ravel()
# print(y.shape)
n_clusters = 3

N = data.shape[0]
avg_N = N // n_clusters
K = avg_N // 10
K = min(K, 10)
K = max(K, 3)
print('K',K)

adj,W_NE = getGraph(data,K)

# Dimension Reduction
if features.shape[0] > dim and features.shape[1] > dim:
    pca = PCA(n_components=dim)
    features = pca.fit_transform(features)
else:
    var = np.var(features, axis=0)
    min_var = np.sort(var)[-1 * dim]
    features = features.T[var >= min_var].T
    features = features[:, :dim]
print('Shape after transformation:', features.shape)
feature = (features - np.mean(features)) / (np.std(features))

# clustering
DBI_list = []
DBI_list.append(10)
nmi_list = []
nmi_list.append(0)
silhouette = []
silhouette.append(-1)
label = zeros([60,adj.shape[0]])
rep = 10

adj_normalized = preprocess_adj(adj)
adj_normalized = (sp.eye(adj_normalized.shape[0]) + adj_normalized)/2
wne_normalized = preprocess_adj(W_NE)
total_dist = []
tt = 0
while 1:
    tt = tt + 1
    power = tt
    dbi = np.zeros(rep)
    nmi = np.zeros(rep)
    silhou = np.zeros(rep)

    feature_w = wne_normalized.dot(feature)
    feature_adj = adj_normalized.dot(feature)
    feature = (feature_adj+feature_w)/2
    feature = (feature - np.mean(feature)) / (np.std(feature))

    u, s, v = sp.linalg.svds(feature, k=n_clusters, which='LM')

    for i in range(rep):
        kmeans = KMeans(n_clusters=n_clusters).fit(u)
        predict_labels = kmeans.predict(u)
        dbi[i] = metrics.davies_bouldin_score(feature,predict_labels)
        nmi[i] = metrics.normalized_mutual_info_score(y, predict_labels)
        silhou[i] = metrics.silhouette_score(feature,predict_labels)

    label[tt] = predict_labels
    dbimean = round(np.mean(dbi),3)
    nmi_means = round(np.mean(nmi),3)
    sil_means = round(np.mean(silhou),3)

    DBI_list.append(dbimean)

    print('power: {}'.format(power),
          'dbimean:{}'.format(dbimean),
          'nmi_mean: {}'.format(nmi_means),
          'silhou_mean:{}'.format(sil_means))
    if DBI_list[tt] > DBI_list[tt - 1] or tt > max_iter:
        print('bestpower: {}'.format(tt - 1))
        pre_labels = label[tt-1]
        # pd.DataFrame(pre_labels).to_csv('pre_label/yan.csv', index=False, header=None)
        score = performance(y, pre_labels)
        break

end = time()
print('running time is :%s seconds'%(end - start))
