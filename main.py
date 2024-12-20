# -*-coding: utf-8 -*-
# @Time    : 2024/10/22 10:25
# @File    : main.py
# @Software: PyCharm


import argparse
import execution
import torch as th
# 参考思路：https://zhuanlan.zhihu.com/p/690835152


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='training或inference', type=str, default='training')

    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--batch_size", help="batch的大小", type=int, default=128)
    parser.add_argument("--epoch", help="epoch的大小", type=int, default=40)
    parser.add_argument("--stop_num", help="early stop机制的触发次数", type=int, default=20)
    parser.add_argument("--class_num", help="类别数目", type=int, default=1)
    parser.add_argument("--input_dim", help="画像特征的维数(含标签)", type=int, default=340)
    parser.add_argument("--input_dimn", help="GNN节点的输入维数", type=int, default=404)
    parser.add_argument("--end_pos", help="序列特征的起始位置(0开始)", type=int, default=341)
    parser.add_argument("--max_len", help="序列最大长度", type=int, default=1500)
    parser.add_argument("--pos_dim", help="序列位置编码的维数", type=int, default=64)
    parser.add_argument("--word_size", help="单词表的大小", type=int, default=76)
    parser.add_argument("--word_dim", help="单词的输出的数据维度", type=int, default=64)
    parser.add_argument("--shrinkcircleid_size", help="缩圈节奏id数目", type=int, default=10)
    parser.add_argument("--edge_dim", help="边的特征维数", type=int, default=64)
    parser.add_argument("--out_feats", help="graph transformer输出的数据维度", type=int, default=256)
    parser.add_argument("--feat_dim1", help="第一个隐含层输出的数据维度", type=int, default=128)
    parser.add_argument("--feat_dim2", help="第二个隐含层输出的数据维度", type=int, default=64)
    parser.add_argument("--feat_dim3", help="第三个隐含层输出的数据维度", type=int, default=32)
    parser.add_argument("--num_heads", help="自注意力机制的头数", type=int, default=2)
    parser.add_argument("--weight_pos", help="正样本权值", type=float, default=0.8)
    parser.add_argument("--alpha", help="Focal Loss的参数值(<0.5关注负样本)", type=float, default=1)
    parser.add_argument("--gamma", help="Focal Loss的参数值(>0对难分类样本给予更高的权重)", type=float, default=2)
    parser.add_argument('--device', type=str, default='cuda' if th.cuda.is_available() else 'cpu',
                        help='Device to use for computation (cuda or cpu)')
    parser.add_argument("--space_input_dim", help="地图坐标的维数", type=int, default=3)
    parser.add_argument("--num_layers", help="transformer模型的层数", type=int, default=2)
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='/models/wedo/ai/seqgraphmodel/')

    parser.add_argument('--data_output', help='Output file path', type=str, default='')
    parser.add_argument('--data_input', help='map_feature_file', type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "training":
        print("进入训练模式!")
        execution.train(args)
    else:
        print("进入推理模式!")
        execution.inference_emebdding(args)