
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class GATModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)

        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1)  # [N, F, T]
        x = x.permute(0, 2, 1)  # [N, T, F]
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        # return self.fc_out(hidden).squeeze()
        return hidden


# from xlstm import (
#     xLSTMBlockStack,
#     xLSTMBlockStackConfig,
#     mLSTMBlockConfig,
#     mLSTMLayerConfig,
#     sLSTMBlockConfig,
#     sLSTMLayerConfig,
#     FeedForwardConfig,
# )
#
# cfg = xLSTMBlockStackConfig(
#     mlstm_block=mLSTMBlockConfig(
#         mlstm=mLSTMLayerConfig(
#             conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
#         )
#     ),
#     slstm_block=sLSTMBlockConfig(
#         slstm=sLSTMLayerConfig(
#             # backend="cuda",
#             backend="vanilla",
#             num_heads=4,
#             conv1d_kernel_size=4,
#             bias_init="powerlaw_blockdependent",
#         ),
#         feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
#     ),
#     context_length=256,
#     num_blocks=7,
#     embedding_dim=128,
#     slstm_at=[1],
#
# )


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
        # self.rnn = klass(
        #     input_size=self.hid_size,
        #     hidden_size=self.hid_size,
        #     num_layers=self.rnn_layer,
        #     batch_first=True,
        #     dropout=self.dropout,
        # )
        # self.rnn = SparseTSF()
        # self.rnn = SegRNN()
        # self.rnn = LTC(self.hid_size, self.hid_size)

        # self.rnn = Kansformer(
        #     d_feat=self.input_size,
        #     d_model=self.hid_size,
        #     nhead=4,
        #     num_layers=self.rnn_layer,
        # )

        # self.rnn = xLSTMBlockStack(cfg)

        # x = torch.randn(4, 256, 128).to("cuda")
        # xlstm_stack = xlstm_stack.to("cuda")
        # y = xlstm_stack(x)

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

        # 特征注意力
        # self.att_net_feature = nn.Sequential()
        # self.att_net_feature.add_module(
        #     "att_fc_in_feature",
        #     nn.Linear(in_features=60, out_features=30),
        # )
        # self.att_net_feature.add_module("att_dropout_feature", torch.nn.Dropout(self.dropout))
        # self.att_net_feature.add_module("att_act_feature", nn.Tanh())
        # self.att_net_feature.add_module(
        #     "att_fc_out_feature",
        #     nn.Linear(in_features=30, out_features=1, bias=False),
        # )
        # self.att_net_feature.add_module("att_softmax_feature", nn.Softmax(dim=1))
        # self.fc_feature = nn.Linear(in_features=60, out_features=self.hid_size)

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

        # out = self.fc_out(rnn_out[:, -1, :])
        return out

        ###### rnn_out_feature = rnn_out.permute(0, 2, 1)
        ###### attention_score_feature = self.att_net_feature(rnn_out_feature)  # [batch, hidden_size, 1]
        ###### out_att_feature = torch.mul(rnn_out_feature, attention_score_feature)
        ###### out_att2_feature = torch.sum(out_att_feature, dim=1)
        ###### out_att2_feature = self.fc_feature(out_att2_feature)
        ######
        ###### out = self.fc_out(
        ######     torch.cat((rnn_out[:, -1, :], out_att2, out_att2_feature), dim=1)
        ###### )  # [batch, seq_len, num_directions * hidden_size] -> [batch, hidden_size]




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
        # print("score.shape")
        # print(score.shape)
        f_1 = torch.matmul(score, self.weight_u).reshape(self.num_heads, 1, -1)
        # print("torch.matmul(score, self.weight_u).shape")
        # print(torch.matmul(score, self.weight_u).shape)
        f_2 = torch.matmul(score, self.weight_v).reshape(self.num_heads, -1, 1)
        # print("f_1.shape")
        # print(f_1.shape)
        # print("f_2.shape")
        # print(f_2.shape)
        logits = f_1 + f_2
        # print("logits.shape")
        # print(logits.shape)
        weight = self.leaky_relu(logits)  # [num_heads, stockNum, stockNum]
        # print("weight.shape")
        # print(weight.shape)

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



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, top_k, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(top_k, d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # print("MultiHeadAttention:", q.shape, k.shape, v.shape)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn, mask_k = self.attention(q, k, v)
        # print(q.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # print(q.shape)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn, mask_k



