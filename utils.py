import torch
import pandas as pd
import numpy as np

def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)

def mae(pred, label):
    loss = (pred - label).abs()
    return torch.mean(loss)

def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity


# from scipy.stats import rankdata
# def cal_cos_similarity(X, x):
#     # X 是一个 N x H 的数据矩阵
#     N, H = X.shape
#
#     # 对每列（特征）进行秩转换
#     ranks = np.apply_along_axis(rankdata, axis=0, arr=X)
#
#     # 计算秩矩阵的皮尔逊相关系数矩阵（也即是斯皮尔曼相关系数矩阵）
#     corr_matrix = np.corrcoef(ranks, rowvar=True)  # 计算行之间的皮尔逊相关系数
#
#     return corr_matrix


def cal_convariance(x, y): # the 2nd dimension of x and y are the same
    e_x = torch.mean(x, dim = 1).reshape(-1, 1)
    e_y = torch.mean(y, dim = 1).reshape(-1, 1)
    e_x_e_y = e_x.mm(torch.t(e_y))
    x_extend = x.reshape(x.shape[0], 1, x.shape[1]).repeat(1, y.shape[0], 1)
    y_extend = y.reshape(1, y.shape[0], y.shape[1]).repeat(x.shape[0], 1, 1)
    e_xy = torch.mean(x_extend*y_extend, dim = 2)
    return e_xy - e_x_e_y


def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level =0).drop('datetime', axis = 1)
        
    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/(x.label>0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return precision, recall, ic, rank_ic
