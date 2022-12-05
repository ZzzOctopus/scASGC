from time import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

def performance(y,y_pred):
    NMI = metrics.normalized_mutual_info_score(y, y_pred)
    print("NMI:", round(NMI,3))
    ARI = metrics.adjusted_rand_score(y, y_pred)
    print("ARI:", round(ARI,3))
    homogeneity = metrics.homogeneity_score(y, y_pred)
    print("homogeneity:", round(homogeneity,3))
    completeness = metrics.completeness_score(y, y_pred)
    print("completeness:", round(completeness,3))


n_cluster = 6
dim = 200
start = time()
data = pd.read_csv('data/deng.csv',header=None)
data = np.array(data)
data = np.log2(1 + data)     #对数化
# 读取真实标签
y = pd.read_csv('data/deng_true_labs.csv',header=None,low_memory=False)
y = np.array(y)
y = y.ravel()
# print(y,y.shape)

pca = PCA(n_components = dim)
X_low = pca.fit_transform(data)
print(X_low.shape)
clustering = KMeans(n_clusters=n_cluster).fit(X_low)
pred = clustering.predict(X_low)
end = time()
score = performance(y,pred)
print('running time is :%s seconds'%(end - start))