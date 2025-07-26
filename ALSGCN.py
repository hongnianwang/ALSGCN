import torch
import torch.nn as nn
import torch.nn.functional as F
from new_module import *


class RankingPredictionModule(nn.Module):
    def __init__(self, hidden_size):
        super(RankingPredictionModule, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    def forward(self, h_concat):
        # 第一个全连接层后接ReLU激活函数
        x = F.relu(self.fc1(h_concat))
        # 第二个全连接层输出排名分数
        y_t_plus_1 = self.fc2(x)
        return y_t_plus_1

class DGC(nn.Module):
    def __init__(self):
        super(DGC, self).__init__()

    def forward(self, h, adj):
        """
        h: 时间 t 的节点特征，形状 [N, hidden_size]
        adj: 时间 t 的动态邻接矩阵，形状 [N, N]
        """
        adj_exp = torch.exp(adj)
        lambda_ij_t = adj_exp / adj_exp.sum(dim=1, keepdim=True)
        h_dt_hat = torch.matmul(lambda_ij_t, h)

        return h_dt_hat



class SGC(nn.Module):
    def __init__(self, hidden_size, dropout, beta, gdep=1):
        super(SGC, self).__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.gdep = gdep
        self.dropout = dropout
        self.beta = beta

    def forward(self, x, adj):
        # adj添加自环并计算每个节点的度
        eye_matrix = torch.diag(torch.ones(adj.size(0), device=x.device))
        adj = adj + eye_matrix
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        ax = torch.mm(a, x)
        for i in range(self.gdep):
            h = self.beta*x + (1-self.beta)*ax
        ho = self.mlp(h)
        return ho


class ALSGCN(nn.Module):
    def __init__(self, nnodes, embedding_size, beta, k=3, alpha=3, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, v=4):
        super(ALSGCN, self).__init__()

        self.feature_extraction = ALSTMModel(
            d_feat=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        self.feature_extraction_SGC = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.emb1 = nn.Embedding(nnodes, embedding_size)
        self.emb2 = nn.Embedding(nnodes, embedding_size)
        self.lin1 = nn.Linear(embedding_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.SGC = SGC(hidden_size, dropout, beta)
        self.DGC = DGC()
        self.prediction = RankingPredictionModule(hidden_size)
        self.k = k
        self.alpha = alpha
        self.d_feat = d_feat

        self.v = v
        self.sub_lin = nn.Linear(hidden_size, 4*hidden_size)  # 利用MLP进行升维
        self.self_attention = SelfAttention(hidden_size, 2*hidden_size)

        self.gat_multiheads = GAT_MultiHeads(
            in_features=hidden_size,
            out_features=hidden_size
        )
        self.gat_multiheads.reset_parameters()

        self.D_DGC = DGC()
        self.S_DGC = DGC()
        self.DS_DGC_linear = nn.Linear(2 * hidden_size, hidden_size)


    def forward(self, x, idx, market_values):
        '''
        use GPU or CPU
        '''

        device = torch.device(torch.get_device(x))
        # device = 'cpu'
        '''
        feature extraction
        '''
        x_dynamic = x.reshape(x.shape[0], self.d_feat, -1)  # [N, F, T]
        # xx = x_dynamic
        x_dynamic = x_dynamic  # [N, T, F]

        h = self.feature_extraction(x_dynamic)

        '''
        Dynamic module
        '''
        dynamic_stock_to_stock = cal_cos_similarity(h, h)  # 余弦相似度

        '''
        Static module
        '''
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)
        # 市值加权 #
        weights = market_values / market_values.sum()
        nodevec1 = nodevec1 * weights.unsqueeze(1)
        nodevec2 = nodevec2 * weights.unsqueeze(1)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        static_stock_to_stock = F.relu(torch.tanh(self.alpha * a))


        concat_res = self.gat_multiheads(h, dynamic_stock_to_stock, static_stock_to_stock)

        '''
        predict
        '''
        y_predict = self.prediction(concat_res)
        y_predict = y_predict.view(-1)
        return y_predict


