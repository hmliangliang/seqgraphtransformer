# -*-coding: utf-8 -*-
# @Time    : 2024/10/22 10:25
# @File    : model.py
# @Software: PyCharm
import torch
import torch as th
import dgl
from dgl.nn import GATConv
import dgl.function as fn
import numpy as np
import random

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)  # 如果有多个 GPU
# 设置 NumPy 随机种子
np.random.seed(3407)
# 设置 Python 的随机种子
random.seed(3407)


# 定义GLU单元网络
class GLUNetwork(th.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLUNetwork, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.bn_layer = th.nn.BatchNorm1d(output_dim)
        self.linear_a = th.nn.Linear(output_dim, output_dim)  # 第一个线性层
        self.linear_b = th.nn.Linear(output_dim, output_dim)  # 第二个线性层

    def forward(self, x):
        x = self.linear(x)
        x = self.bn_layer(x)
        a = self.linear_a(x)  # 通过第一个线性层
        b = self.linear_b(x)  # 通过第二个线性层
        x = a * torch.sigmoid(b)  # GLU 计算
        return x


'''
# 定义MLP单元网络
class GLUNetwork(th.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLUNetwork, self).__init__()
        self.linear = th.nn.Linear(input_dim, output_dim)
        self.bn_layer = th.nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn_layer(x)
        x = th.nn.functional.leaky_relu(x)
        return x
'''

'''
# 实现的是SwiGLU单元
class GLUNetwork(th.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLUNetwork, self).__init__()
        self.linear_a = th.nn.Linear(input_dim, output_dim, bias=False)  # 第一个线性层
        self.linear_b = th.nn.Linear(input_dim, output_dim, bias=False)  # 第二个线性层
        self.bn_a = th.nn.BatchNorm1d(input_dim)
        self.linear_c = th.nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.bn_a(x)
        a = self.linear_a(x)  # 通过第一个线性层
        b = self.linear_b(x)  # 通过第二个线性层
        x = self.linear_c(a * th.nn.functional.silu(b))
        return x  # SwiGLU 计算
'''

'''
# 定义GEGLU单元
class GLUNetwork(th.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLUNetwork, self).__init__()
        self.linear1 = th.nn.Linear(input_dim, output_dim, bias=False)
        self.linear2 = th.nn.Linear(input_dim, output_dim, bias=False)
        self.linear3 = th.nn.Linear(output_dim, output_dim, bias=False)
        self.bn_layer = th.nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = self.bn_layer(x)
        gate = th.sigmoid(self.linear1(x))  # 计算门控
        output = self.linear2(x)  # 计算线性变换
        x = gate * output + th.nn.functional.elu(output)  # GEGLU 公式
        return x
'''


class Model(th.nn.Module):
    def __init__(self, word_size=76, word_dim=64, input_dim=404, out_feat=256, edge_dim=40,
                 feat_dim1=128, feat_dim2=64, feat_dim3=32, class_num=2, num_heads=2, num_layers=1):
        super(Model, self).__init__()
        self.word_size = word_size
        self.word_dim = word_dim
        self.out_feat = out_feat
        self.word_embedding = th.nn.Embedding(word_size, word_dim)
        self.gat_layer = GATConv(input_dim, out_feat, num_heads=num_heads, allow_zero_in_degree=True,
                                 activation=th.nn.functional.elu) # (n, num_heads, out_feat)

        self.layers_edge = GLUNetwork(word_dim, word_dim)
        self.lstm_layer = th.nn.LSTM(out_feat + edge_dim, out_feat + edge_dim, num_layers, batch_first=True)
        self.mlp_layer = th.nn.Linear(out_feat + edge_dim, out_feat + edge_dim)

        self.layers1 = GLUNetwork(out_feat + edge_dim, feat_dim1)
        self.layers2 = GLUNetwork(feat_dim1, feat_dim2)
        self.layers3 = GLUNetwork(feat_dim2, feat_dim3)
        self.layers4 = th.nn.Linear(feat_dim3, class_num)

        '''
        self.transformer_bn = th.nn.BatchNorm1d(out_feat + edge_dim)

        self.profile_layer = th.nn.Linear(out_feat, out_feat + edge_dim)
        self.prelu_profile = th.nn.PReLU()
        '''

        '''
        self.layers1 = th.nn.Linear(out_feat + edge_dim, feat_dim1)
        self.batch_layer1 = th.nn.BatchNorm1d(feat_dim1)
        self.layers2 = th.nn.Linear(feat_dim1, feat_dim2)
        self.batch_layer2 = th.nn.BatchNorm1d(feat_dim2)
        self.layers3 = th.nn.Linear(feat_dim2, feat_dim3)
        self.batch_layer3 = th.nn.BatchNorm1d(feat_dim3)
        self.layers4 = th.nn.Linear(feat_dim3, class_num)
        '''

    def forward(self, g, device):
        id = th.tensor([i for i in range(self.word_size)], dtype=th.long).to(device)
        word_embedding = self.word_embedding(id % self.word_size)
        # 此时输入特征变为 n * input_dimn, input_dimn = input_dimn + word_dim
        m = int(g.ndata['h'].shape[0] / self.word_size)
        word_embedding = word_embedding.repeat(m, 1)
        g.ndata['h'] = th.cat([g.ndata['h'], word_embedding], dim=1)
        data = self.gat_layer(g, g.ndata['h'])
        data = th.mean(data, dim=1)
        # 边的特征聚合
        with g.local_scope():
            # 定义消息传递函数，将边特征作为消息传递
            g.update_all(fn.copy_e('e', 'm'), fn.mean('m', 'agg_ef'))

            # 获取聚合后的边特征
            agg_ef = g.ndata['agg_ef'] # 边的特征edge_dim维
            # 边特征激活函数与GLU计算
            agg_ef = self.layers_edge(agg_ef)

            # 拼接节点特征和聚合后的边特征
            data = th.cat([data, agg_ef], dim=1) # out_feat + edge_dim
        g.ndata['h'] = data
        # 计算筛选后节点的特征平均值
        # data = dgl.mean_nodes(g, 'h')

        # 将批量图拆分为单个子图
        subgraphs = dgl.unbatch(g)
        data = torch.zeros(len(subgraphs), g.ndata['h'].shape[1]).to(device)
        flag = 0
        for gs in subgraphs:
            # 计算每个节点的度
            in_degrees = gs.in_degrees() + gs.out_degrees()

            # 筛选出度大于0的节点
            mask = in_degrees > 0
            filtered_feats = gs.ndata['h'][mask]

            # 计算筛选后节点的特征平均值
            if filtered_feats.shape[0] > 0:
                filtered_feats = th.unsqueeze(filtered_feats, dim=0)
                # LSTM融合样本
                filtered_feats, _ = self.lstm_layer(filtered_feats)
                filtered_feats = th.squeeze(filtered_feats, dim=0)
                mean_feat = filtered_feats.mean(dim=0)
            else:
                # mean_feat = torch.zeros(g.ndata['h'].shape[1], requires_grad=True).to(device)
                filtered_feats = gs.ndata['h']
                indices = th.randperm(filtered_feats.shape[0])
                filtered_feats = filtered_feats[indices]

                filtered_feats = self.mlp_layer(filtered_feats)
                filtered_feats = th.nn.functional.leaky_relu(filtered_feats)
                mean_feat = filtered_feats.mean(dim=0)
            data[flag, :] = mean_feat
            flag += 1

        data = self.layers1(data)
        data = self.layers2(data)
        data = self.layers3(data)
        data_embedding = data
        data = self.layers4(data)
        data = torch.sigmoid(data)

        '''
        data = self.layers1(data)
        data = self.batch_layer1(data)
        data = th.sigmoid(data)
        data = self.layers2(data)
        data = self.batch_layer2(data)
        data = th.sigmoid(data)
        data = self.layers3(data)
        data = self.batch_layer3(data)
        data = th.sigmoid(data)
        data = self.layers4(data)
        data = torch.sigmoid(data)
        '''

        return [data_embedding, data]