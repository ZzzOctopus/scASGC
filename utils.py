"""
The implementation of Network Enhancement (NE) is modified from
https://github.com/wangboyunze/Network_Enhancement on 2021.11.11,
which is released under GNU General Public License v3.0.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import metrics

def Proprecess(data):
    # data preprocessing, including the removal of genes expressed in less than 95% of cells and logarithm
    X = np.array(data)
    print('raw shape:',X.shape)
    X = X.T
    cell = []
    for i in range(len(X)):
        cell.append(len(np.argwhere(X[i] > 0)))
    cell = np.array(cell)
    t = int(0.05 * X.shape[1])
    index = np.argwhere(cell > t)
    index = index.reshape(len(index))
    X = X[index]
    X = np.transpose(X)
    print("after proprecessing", X.shape)
    X = np.log2(1 + X)
    return X

def normalization(X):
    # data normalization
    X =np.array(X)
    print(X.shape)
    for i in range(len(X)):
        X[i] = X[i] / sum(X[i]) * 100000
    X = np.log2(X + 1)
    return X

def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn

def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix

def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W

def getNeMatrix(W_in):
    N = len(W_in)
    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20

    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2

    DD = np.sum(np.abs(W0), axis=0)

    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))

    P = TransitionFields(P, N, eps)

    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T

    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W

def getGraph(X,K):
    # Construct cell graph
    co_matrix = np.corrcoef(X)
    X = normalization(X)
    in_matrix = np.corrcoef(X)
    # pd.DataFrame(co_matrix).to_csv('ting_CO.csv', index=False, header=None)
    NE_matrix = getNeMatrix(in_matrix)
    # pd.DataFrame(NE_matrix).to_csv('ting_NE.csv', index=False, header=None)

    data = NE_matrix.reshape(-1)
    data = np.sort(data)
    data = data[:-int(len(data) * 0.02)]

    min_sh = data[0]
    max_sh = data[-1]

    delta = (max_sh - min_sh) / 100

    temp_cnt = []
    for i in range(20):
        s_sh = min_sh + delta * i
        e_sh = s_sh + delta
        temp_data = data[data > s_sh]
        temp_data = temp_data[temp_data < e_sh]
        temp_cnt.append([(s_sh + e_sh) / 2, len(temp_data)])

    candi_sh = -1
    for i in range(len(temp_cnt)):
        pear_sh, pear_cnt = temp_cnt[i]
        if 0 < i < len(temp_cnt) - 1:
            if pear_cnt < temp_cnt[i + 1][1] and pear_cnt < temp_cnt[i - 1][1]:
                candi_sh = pear_sh
                break
    if candi_sh < 0:
        for i in range(1, len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if pear_cnt * 2 < temp_cnt[i - 1][1]:
                candi_sh = pear_sh
    if candi_sh == -1:
        candi_sh = 0.3

    propor = len(NE_matrix[NE_matrix <= candi_sh]) / (len(NE_matrix) ** 2)
    propor = 1 - propor
    thres = np.sort(NE_matrix)[:, -int(len(NE_matrix) * propor)]
    co_matrix.T[NE_matrix.T <= thres] = 0

    up_K = np.sort(co_matrix)[:, -K]
    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1
    mat_K = (mat_K+mat_K.T)/2
    mat_K[mat_K>=0.5] = 1
    W_NE = mat_K*co_matrix
    # pd.DataFrame(W_NE).to_csv('NE.csv', index=False, header=None)
    # pd.DataFrame(mat_K).to_csv('graph.csv', index=False, header=None)
    return mat_K,W_NE

def normalize_adj(adj):
    # Graph convolution
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj, loop=True):
    if loop:
        adj = adj + sp.eye(adj.shape[0])
    adj_normalized = normalize_adj(adj)
    return adj_normalized

def performance(y,y_pred):
    # Cluster metrics
    NMI = metrics.normalized_mutual_info_score(y, y_pred)
    print("NMI:", round(NMI,3))
    ARI = metrics.adjusted_rand_score(y, y_pred)
    print("ARI:", round(ARI,3))
    homogeneity = metrics.homogeneity_score(y, y_pred)
    print("homogeneity:", round(homogeneity,3))
    completeness = metrics.completeness_score(y, y_pred)
    print("completeness:", round(completeness,3))
    return NMI