# -*-coding: utf-8 -*-
# @Time    : 2024/10/22 10:27
# @File    : execution.py
# @Software: PyCharm


import os
import datetime

import torch as th
import torch.optim as optim
import pandas as pd
import numpy as np
import dgl
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import confusion_matrix
from model import Model
from concurrent.futures import ThreadPoolExecutor
import random

th.manual_seed(3407)
if th.cuda.is_available():
    th.cuda.manual_seed(3407)
    th.cuda.manual_seed_all(3407)  # 如果有多个 GPU
# 设置 NumPy 随机种子
np.random.seed(3407)
# 设置 Python 的随机种子
random.seed(3407)


# 定义Focal Loss函数
class FocalLoss(th.nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', epsilon=1e-6, epsilon_smoth=0.005):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = epsilon
        self.epsilon_smoth = epsilon_smoth

    def forward(self, inputs, targets):
        # inputs 是正类的预测概率
        p_t = inputs  # 直接使用预测概率
        # 标签平滑
        targets = targets * (1 - self.epsilon_smoth) + (1 - targets) * self.epsilon_smoth

        # 裁剪预测概率
        p_t = th.clamp(p_t, self.epsilon, 1 - self.epsilon)

        # 计算 Focal Loss
        F_loss = -self.alpha * (1 - p_t) ** self.gamma * (targets * th.log(p_t) + (1 - targets) * th.log(1 - p_t))

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class GraphDataset(th.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __getitem__(self, idx):
        return self.graphs, self.labels

    def __len__(self):
        # self.graphs.num_graphs
        return 1


def get_positional_encoding(max_seq_len, d_model):
    pos = th.arange(max_seq_len, dtype=th.float32).unsqueeze(1)
    i = th.arange(d_model, dtype=th.float32).unsqueeze(0)
    angle_rates = 1 / th.pow(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    # Apply sin to even indices in the array; 2i
    sines = th.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array; 2i+1
    cosines = th.cos(angle_rads[:, 1::2])
    pos_encoding = th.zeros((max_seq_len, d_model))
    pos_encoding[:, 0::2] = sines
    pos_encoding[:, 1::2] = cosines
    # 线性加权
    # weights = th.linspace(1, max_seq_len, max_seq_len).unsqueeze(1)  # 权重从1到max_seq_len线性增加
    # 指数加权
    weights = th.exp(th.linspace(0, 6.5, max_seq_len)).unsqueeze(1)
    pos_encoding *= weights  # 对位置编码进行加权

    return pos_encoding


# Xavier 初始化函数
def init_weights(m):
    if isinstance(m, th.nn.Linear):
        # 使用 Xavier 正态分布初始化
        th.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            th.nn.init.zeros_(m.bias)


'''
# Kaiming 正态分布初始化
def init_weights(m):
    if isinstance(m, th.nn.Linear):
        th.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            th.nn.init.zeros_(m.bias)
'''

def build_graph(i, profile_data, data, pos_vec, args):
    sample_profile_data = th.reshape(profile_data[i, :-1], (1, -1))
    sample_profile_label = th.reshape(profile_data[i, -1], (1, -1))
    sample_actiontype_id = th.tensor(data.iloc[i, 0], dtype=th.long)
    # sample_shrinkcircleid_id = th.tensor(data.iloc[i, 1], dtype=th.float32)
    # sample_posxyz_id = th.sigmoid(th.tensor(data.iloc[i, 2:5].values.tolist(), dtype=th.float32))

    g = dgl.graph([], num_nodes=args.word_size)
    edges_num = sample_actiontype_id.shape[0] - 1
    g.ndata['h'] = th.zeros(args.word_size, args.end_pos - 1) + th.sigmoid(sample_profile_data)  # 340维
    src_id = sample_actiontype_id[0:-1]
    dst_id = sample_actiontype_id[1:]
    g.add_edges(src_id, dst_id)
    g.edata['e'] = pos_vec[th.tensor([i for i in range(edges_num)], dtype=th.long) % args.max_len, :]  # 40维
    return g, sample_profile_label


def train(args):
    model = Model(word_size=args.word_size, word_dim=args.word_dim, input_dim=args.input_dimn, out_feat=args.out_feats,
                  edge_dim=args.edge_dim, feat_dim1=args.feat_dim1, feat_dim2=args.feat_dim2, feat_dim3=args.feat_dim3,
                  class_num=args.class_num, num_heads=args.num_heads, num_layers=args.num_layers).to(args.device)
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 读取数据
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    # 获取位置编码信息
    pos_vec = get_positional_encoding(args.max_len, args.pos_dim)
    epoch_num = 0
    auc_best = 0
    print("开始读取训练数据! {}".format(datetime.datetime.now()))
    with ThreadPoolExecutor() as executor:
        for epoch in range(args.epoch):
            count = 0
            epoch_num += 1
            for file in input_files:
                count += 1
                print("一共{}个文件,当前正在处理第{}个训练文件,文件路径:{}......".format(len(input_files), count,
                                                                                        os.path.join(path, file)))
                # 读取训练数据
                data = pd.read_csv(os.path.join(path, file), sep=';', header=None).astype(object)
                n_samples, _ = data.shape
                '''
                训练集特征：
                0-339: 玩家的画像特征
                340: 次局开局标签

                341: actiontype
                342: shrinkcircleid
                343-345: x,y,z
                '''
                # 画像+标签数据
                profile_data = th.tensor(data.iloc[:, 0:args.end_pos].values.astype(np.float), dtype=th.float32)
                data = data.iloc[:, args.end_pos:]
                data = data.applymap(lambda x: list(map(float, x.split(','))))
                graphs = []
                futures = [executor.submit(build_graph, i, profile_data, data, pos_vec, args) for i in range(n_samples)]
                for future in futures:
                    graphs.append(future.result())
                del futures
                # 将多个图组合成一个批次
                batched_graph = dgl.batch([g for g, _ in graphs])  # 这里将多个图组合成一个批次
                labels = th.cat([label for _, label in graphs])  # 合并标签
                dataset = GraphDataset(batched_graph, labels)
                dataloader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                for batched_graph, labels in dataloader:
                    batched_graph = batched_graph.to(args.device)
                    labels = labels.squeeze(0).to(args.device)
                    model.train()

                    criterion = th.nn.BCELoss(
                        weight=th.where(labels >= 1.0, th.tensor(args.weight_pos, dtype=th.float32).to(args.device),
                                        th.tensor(1.0, dtype=th.float32).to(args.device)))

                    # criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, reduction='mean')
                    _, pred = model(batched_graph, args.device)
                    loss = criterion(pred, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                auc_val = test_model(model, args)
                print("第{}个epoch第{}个文件的验证集测试AUC结果为:{} 历史最佳AUC:{}  {}".format(
                    epoch, count, auc_val, auc_best, datetime.datetime.now()))
                if auc_val > auc_best:
                    auc_best = auc_val
                    # 保存模型
                    os.makedirs("./seqgraphmodel", exist_ok=True)
                    th.save(model.state_dict(), './seqgraphmodel/seqgraphmodel.pth')
                    cmd = "s3cmd put -r ./seqgraphmodel " + args.model_output
                    os.system(cmd)
                    epoch_num = 0
                    if epoch_num > args.stop_num:
                        break


# 验证模型效果
def test_model(model, args):
    model.eval()
    # 读取数据
    # 位置编码
    pos_vec = get_positional_encoding(args.max_len, args.pos_dim)
    model.eval()
    path = args.data_input.split(',')[1]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    count = 0
    tag = True
    with ThreadPoolExecutor() as executor:
        for file in input_files:
            count += 1
            # 读取训练数据
            data = pd.read_csv(os.path.join(path, file), sep=';', header=None).astype(object)
            n_samples, _ = data.shape
            '''
            验证集剩余特征：
            0-339: 玩家的画像特征
            340: 是否次局标签

            341: actiontype
            342: shrinkcircleid
            343-345: x,y,z

            '''
            # 画像+标签数据
            profile_data = th.tensor(data.iloc[:, 0:args.end_pos].values.astype(np.float), dtype=th.float32)
            data = data.iloc[:, args.end_pos:]
            data = data.applymap(lambda x: list(map(float, x.split(','))))
            graphs = []
            futures = [executor.submit(build_graph, i, profile_data, data, pos_vec, args) for i in range(n_samples)]
            for future in futures:
                graphs.append(future.result())
            del futures
            batched_graph = dgl.batch([g for g, _ in graphs])  # 这里将多个图组合成一个批次
            labels = th.cat([label for _, label in graphs])  # 合并标签
            dataset = GraphDataset(batched_graph, labels)
            dataloader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            for batched_graph, labels in dataloader:
                batched_graph = batched_graph.to(args.device)
                labels = labels.squeeze(0).to(args.device)
                with th.no_grad():
                    _, res = model(batched_graph, args.device)
                if tag:
                    result_all = res
                    label_all = labels
                    tag = False
                else:
                    result_all = th.cat([result_all, res], dim=0)
                    label_all = th.cat([label_all, labels], dim=0)
        if args.device == "cuda":
            label_all = label_all.cpu()
            result_all = result_all.cpu()
        auc_val = roc_auc_score(label_all.numpy(), result_all.detach().numpy())
        cmx = confusion_matrix(label_all.numpy(), (result_all > 0.5).float().detach().numpy())
        print("混淆矩阵:", cmx)
        return auc_val


# 推理模型过程
def inference_emebdding(args):
    # 定义模型
    model = Model(word_size=args.word_size, word_dim=args.word_dim, input_dim=args.input_dimn, out_feat=args.out_feats,
                  edge_dim=args.edge_dim, feat_dim1=args.feat_dim1, feat_dim2=args.feat_dim2, feat_dim3=args.feat_dim3,
                  class_num=args.class_num, num_heads=args.num_heads, num_layers=args.num_layers).to(args.device)
    # 加载模型的权值
    cmd = "s3cmd get -r  " + args.model_output + "seqgraphmodel"
    os.system(cmd)
    model.load_state_dict(th.load('seqgraphmodel/seqgraphmodel.pth', map_location=th.device(args.device)))

    pos_vec = get_positional_encoding(args.max_len, args.pos_dim)
    model.eval()
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    count = 0
    with ThreadPoolExecutor() as executor:
        for file in input_files:
            count += 1
            # 读取训练数据
            data = pd.read_csv(os.path.join(path, file), sep=';', header=None).astype(object)
            ID = data.iloc[:, 0]
            data = data.iloc[:, 1:]
            n_samples, _ = data.shape
            '''
            验证集剩余特征：
            0-339: 玩家的画像特征
            340: 是否次局标签

            341: actiontype
            342: shrinkcircleid
            343-345: x,y,z

            '''
            # 画像+标签数据
            profile_data = th.tensor(data.iloc[:, 0:args.end_pos].values.astype(np.float), dtype=th.float32)
            data = data.iloc[:, args.end_pos:]
            data = data.applymap(lambda x: list(map(float, x.split(','))))
            graphs = []
            futures = [executor.submit(build_graph, i, profile_data, data, pos_vec, args) for i in range(n_samples)]
            for future in futures:
                graphs.append(future.result())
            del futures
            batched_graph = dgl.batch([g for g, _ in graphs])  # 这里将多个图组合成一个批次
            labels = th.cat([label for _, label in graphs])  # 合并标签
            dataset = GraphDataset(batched_graph, labels)
            dataloader = GraphDataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            for batched_graph, labels in dataloader:
                batched_graph = batched_graph.to(args.device)
                labels = labels.squeeze(0).to(args.device)
                with th.no_grad():
                    res, _ = model(batched_graph, args.device)
                    result_all = res
            if args.device == "cuda":
                result_all = result_all.cpu()
            result = np.zeros((n_samples, 33)).astype(dtype=str)
            result[:, 0] = ID.values.astype(str)
            result[:, 1::] = result_all.numpy().astype(str)
            output_file = os.path.join(args.data_output, 'pred_{}.csv'.format(count))

            # 使用 numpy.savetxt 写入 CSV 文件
            with open(output_file, mode="a") as resultfile:
                # 写入数据
                np.savetxt(resultfile, result, delimiter=',', fmt='%s')  # 使用 %s 以支持字符串和数字
            print("第{}个数据文件已经写入完成,写入数据的行数{} {}".format(count, n_samples, datetime.datetime.now()))