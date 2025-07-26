import torch
import pandas as pd
import numpy as np


def mse(pred, label):
    """
    计算均方误差（Mean Squared Error）。

    参数：
    - pred：预测值
    - label：真实值

    返回：
    - 均方误差
    """
    loss = (pred - label) ** 2
    return torch.mean(loss)


def mae(pred, label):
    """
    计算平均绝对误差（Mean Absolute Error）。

    参数：
    - pred：预测值
    - label：真实值

    返回：
    - 平均绝对误差
    """
    loss = (pred - label).abs()
    return torch.mean(loss)

'''
ALSGCN的loss
'''
def ALSGCN_loss(y_pred, y_true, mu=0.3):
    # 点对点回归损失
    mse_loss = torch.mean((y_pred - y_true) ** 2)

    # 成对排序感知损失（向量化操作）
    diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    pairwise_loss = torch.sum(torch.max(torch.tensor(0.0), -diff_pred * diff_true))

    pairwise_loss *= mu

    # 总损失
    total_loss = mse_loss + pairwise_loss
    return total_loss

def cal_cos_similarity(x, y):
    """
    计算余弦相似度。

    参数：
    - x：输入向量
    - y：输入向量

    返回：
    - 余弦相似度
    """
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
    cos_similarity = xy / x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity

# def cal_cos_similarity(x, y):
#     """
#     计算皮尔逊相关系数。
#
#     参数：
#     - x：输入向量
#     - y：输入向量
#
#     返回：
#     - 皮尔逊相关系数
#     """
#     # 计算均值
#     x_mean = torch.mean(x, dim=1, keepdim=True)
#     y_mean = torch.mean(y, dim=1, keepdim=True)
#
#     # 中心化
#     x_centered = x - x_mean
#     y_centered = y - y_mean
#
#     # 计算分子（协方差）
#     numerator = torch.sum(x_centered * y_centered, dim=1)
#
#     # 计算分母（标准差）
#     x_norm = torch.sqrt(torch.sum(x_centered * x_centered, dim=1))
#     y_norm = torch.sqrt(torch.sum(y_centered * y_centered, dim=1))
#
#     # 计算皮尔逊相关系数
#     pearson_correlation = numerator / (x_norm * y_norm)
#
#     # 避免 NaN（例如除以零的情况）
#     pearson_correlation[pearson_correlation != pearson_correlation] = 0
#
#     return pearson_correlation



def cal_convariance(x, y):
    """
    计算协方差。

    参数：
    - x：输入向量
    - y：输入向量

    返回：
    - 协方差
    """
    e_x = torch.mean(x, dim=1).reshape(-1, 1)
    e_y = torch.mean(y, dim=1).reshape(-1, 1)
    e_x_e_y = e_x.mm(torch.t(e_y))
    x_extend = x.reshape(x.shape[0], 1, x.shape[1]).repeat(1, y.shape[0], 1)
    y_extend = y.reshape(1, y.shape[0], y.shape[1]).repeat(x.shape[0], 1, 1)
    e_xy = torch.mean(x_extend * y_extend, dim=2)
    return e_xy - e_x_e_y


def metric_fn(preds):
    """
    计算评估指标。

    参数：
    - preds：预测结果DataFrame

    返回：
    - 精确度
    - 召回率
    - 信息系数
    - 排名信息系数
    """
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level=0).drop('datetime', axis=1)

    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x: (x.label[:k] > 0).sum() / (x.label > 0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return precision, recall, ic, rank_ic
