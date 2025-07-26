import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ALSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU"):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = nn.GRU(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
            # bidirectional=True  # BiGRU
        )
        self.gru_out = nn.Linear(self.hid_size * 2, self.hid_size)


        self.fc_out = nn.Linear(in_features=self.hid_size * 2, out_features=self.hid_size)
        # self.fc_out = nn.Linear(in_features=self.hid_size, out_features=self.hid_size)
        # 时间注意力
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hid_size, out_features=int(self.hid_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hid_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, x):
        # rnn_out = self.rnn(self.net(x))
        rnn_out, _ = self.rnn(self.net(x))  # [batch, seq_len, num_directions * hidden_size]
        # rnn_out = self.gru_out(rnn_out)

        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att2 = torch.sum(out_att, dim=1)

        out = self.fc_out(
            torch.cat((rnn_out[:, -1, :], out_att2), dim=1)
        )  # [batch, seq_len, num_directions * hidden_size] -> [batch, hidden_size]

        return out


class GAT_MultiHeads(nn.Module):
    '''
    在这里我们采用多头图注意力机制。
    '''

    def __init__(self, in_features,
                 out_features=128,
                 negative_slope=0.3,
                 num_heads=8,
                 bias=True, residual=True):
        super(GAT_MultiHeads, self).__init__()
        self.num_heads = num_heads
        self.out_features = int(out_features / self.num_heads)
        self.weight = nn.Linear(in_features, self.num_heads * self.out_features)
        self.weight_u = nn.Parameter(torch.FloatTensor(self.num_heads, self.out_features, 1))
        self.weight_v = nn.Parameter(torch.FloatTensor(self.num_heads, self.out_features, 1))
        self.weight_cat = nn.Linear(self.num_heads * self.out_features * 2, self.num_heads * self.out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        self.weight_d = nn.Linear(out_features, out_features)
        if self.residual:
            self.project = nn.Linear(in_features, self.num_heads * self.out_features)
        else:
            self.project = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, self.num_heads * self.out_features))
        else:
            self.register_parameter('bias', None)

        # self.kan_dynamic = KAN_linear(out_features, out_features)
        # self.kan_static = KAN_linear(out_features, out_features)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight_u.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight_v.data, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, inputs, Dgraph=None, Sgraph=None, requires_weight=False):
        stockNum = inputs.shape[0]
        score = self.weight(inputs)
        score = score.reshape(stockNum, self.num_heads, self.out_features).permute(dims=(1, 0, 2))
        f_1 = torch.matmul(score, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(score, self.weight_v).reshape(self.num_heads, -1, 1)

        logits = f_1 + f_2
        weight = self.leaky_relu(logits)  # [num_heads, stockNum, stockNum]

        # 使用Dgraph和Sgraph计算加权的注意力值
        Dgraph = Dgraph.unsqueeze(0).repeat((self.num_heads, 1, 1))
        # print("Dgraph.shape")
        # print(Dgraph.shape)
        D_masked_weight = torch.mul(weight, Dgraph).to_sparse()

        D_attn_weights = torch.sparse.softmax(D_masked_weight, dim=2).to_dense()
        D_support = torch.matmul(D_attn_weights, score)
        D_support = D_support.permute(dims=(1, 0, 2)).reshape(stockNum, self.num_heads * self.out_features)
        # D_support = self.kan_dynamic(D_support)

        if Sgraph is not None:
            Sgraph = Sgraph.unsqueeze(0).repeat((self.num_heads, 1, 1))
            S_masked_weight = torch.mul(weight, Sgraph).to_sparse()
            S_attn_weights = torch.sparse.softmax(S_masked_weight, dim=2).to_dense()
            S_support = torch.matmul(S_attn_weights, score)
            S_support = S_support.permute(dims=(1, 0, 2)).reshape(stockNum, self.num_heads * self.out_features)
            # S_support = self.kan_dynamic(S_support)

            # 将D支持和S支持进行拼接
            score = torch.cat([S_support, D_support], dim=1)
            score = self.weight_cat(score)
        else:
            score = self.weight_d(D_support)

        # 添加偏置项和残差连接
        if self.bias is not None:
            score = score + self.bias
        if self.residual:
            score = score + self.project(inputs)
        if requires_weight:
            return score, D_attn_weights, S_attn_weights
        else:
            return score

