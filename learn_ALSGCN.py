import random

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.optim as optim  # 导入 PyTorch 优化器
import os  # 导入 os 模块，用于操作系统相关功能
import copy  # 导入 copy 模块，用于对象的复制
import json  # 导入 json 模块，用于 JSON 格式数据的处理
import argparse  # 导入 argparse 模块，用于命令行参数解析
import datetime  # 导入 datetime 模块，用于日期时间处理
import collections  # 导入 collections 模块，用于集合数据类型操作
import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理
from tqdm import tqdm  # 导入 tqdm 模块，用于进度条显示
import qlib  # 导入 qlib 库，用于量化研究
from qlib.qlib.config import REG_US, REG_CN  # 导入 qlib 配置信息

from qlib.qlib.data.dataset import DatasetH  # 导入 Qlib 数据集模块
from qlib.qlib.data.dataset.handler import DataHandlerLP  # 导入 Qlib 数据处理模块
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 用于可视化
from qlib.qlib.contrib.model.pytorch_gru import GRUModel  # 导入 PyTorch GRU 模型
from qlib.qlib.contrib.model.pytorch_lstm import LSTMModel  # 导入 PyTorch LSTM 模型
from qlib.qlib.contrib.model.pytorch_gats import GATModel  # 导入 PyTorch GAT 模型
from qlib.qlib.contrib.model.pytorch_sfm import SFM_Model  # 导入 PyTorch SFM 模型
from qlib.qlib.contrib.model.pytorch_alstm import ALSTMModel  # 导入 PyTorch ALSTM 模型
from qlib.qlib.contrib.model.pytorch_transformer import Transformer  # 导入 PyTorch Transformer 模型
from model_zhushi import MLP, HIST  # 导入自定义模型
from ALSGCN import ALSGCN
from utils_zhushi import *  # 导入工具函数
from dataloader_zhushi import DataLoader  # 导入数据加载器

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 根据是否有 GPU 设置设备
# device = 'cpu'
print("device=", device)
EPS = 1e-12  # 定义一个极小值，用于避免除零错误

from torch.utils.tensorboard import SummaryWriter


# 根据模型名称返回对应的模型类
def get_model(model_name):
    return ALSGCN

# 对参数列表进行平均
def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params' % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params

'''
ALSGCN模型的loss
'''
def loss_ALSGCN(pred, label, args):
    # mask = ~torch.isnan(label)  # 过滤掉标签值中的NaN值
    # return ALSGCN_loss(pred[mask], label[mask], args.mu)  # 返回均方误差+排序误差
    return loss_fn(pred, label, args)
def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)  # 过滤掉标签值中的NaN值
    return mse(pred[mask], label[mask])  # 返回均方误差

global_log_file = None  # 全局日志文件路径，初始值为None


def pprint(*args):  # 定义打印函数，可以接收任意数量的参数

    # 打印带有 UTC+8 时间的信息
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'  # 获取当前时间并转换为 UTC+8 格式
    print(time, *args, flush=True)  # 打印带有时间的信息，刷新缓冲区

    if global_log_file is None:  # 如果全局日志文件路径为None
        return  # 直接返回，不进行日志记录

    # 将信息写入日志文件
    with open(global_log_file, 'a') as f:  # 打开日志文件，追加模式
        print(time, *args, flush=True, file=f)  # 将带有时间的信息写入日志文件，刷新缓冲区


# 全局变量，用于记录训练步数
global_step = -1


# 训练一个 epoch
def train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix=None):
    global global_step
    model.train()  # 设置模型为训练模式
    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, market_value, stock_index, _ = train_loader.get(slc)
        pred = model(feature, stock_index, market_value)  # 进行前向传播


        loss = loss_ALSGCN(pred, label, args)
        # loss = loss_fn(pred, label, args)

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)  # 梯度裁剪
        optimizer.step()  # 更新模型参数


# 测试一个 epoch
def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, prefix='Test'):
    model.eval()  # 设置模型为评估模式

    losses = []  # 存储损失
    preds = []  # 存储预测结果

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index = test_loader.get(slc)  # 获取数据

        with torch.no_grad():  # 不需要梯度计算
            pred = model(feature, stock_index, market_value)  # 进行前向传播
            loss = loss_ALSGCN(pred, label, args)


            preds.append(
                pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy()}, index=index))  # 将预测结果存储

        losses.append(loss.item())  # 存储损失值

    preds = pd.concat(preds, axis=0)  # 合并预测结果
    precision, recall, ic, rank_ic = metric_fn(preds)  # 计算评价指标
    scores = ic

    writer.add_scalar(prefix + '/Loss', np.mean(losses), epoch)  # 记录损失
    writer.add_scalar(prefix + '/std(Loss)', np.std(losses), epoch)  # 记录损失的标准差
    writer.add_scalar(prefix + '/' + args.metric, scores, epoch)  # 记录评价指标
    # writer.add_scalar(prefix + '/std(' + args.metric + ')', np.std(scores), epoch)  # 记录评价指标的标准差

    return np.mean(losses), scores, precision, recall, ic, rank_ic  # 返回损失和评价指标


# 推断函数，用于使用模型进行预测
def inference(model, data_loader, stock2concept_matrix=None):
    model.eval()  # 设置模型为评估模式

    preds = []  # 存储预测结果
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index = data_loader.get(slc)  # 获取数据
        with torch.no_grad():  # 不需要梯度计算
            pred = model(feature, stock_index)  # 进行前向传播
            preds.append(
                pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy()}, index=index))  # 将预测结果存储

    preds = pd.concat(preds, axis=0)  # 合并预测结果
    return preds  # 返回预测结果


# 创建数据加载器
def create_loaders(args):
    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')  # 解析训练开始日期
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')  # 解析测试结束日期
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')  # 解析训练结束日期

    hanlder = {
        'class': 'Alpha360',
        'module_path': 'qlib.qlib.contrib.data.handler',
        'kwargs':
            {
                'start_time': start_time,
                'end_time': end_time,
                'fit_start_time': start_time,
                'fit_end_time': train_end_time,
                'instruments': args.data_set,
                'infer_processors': [
                    {
                        'class': 'RobustZScoreNorm',
                        'kwargs':
                            {
                                'fields_group': 'feature',
                                'clip_outlier': True
                            }
                    },
                    {
                        'class': 'Fillna',
                        'kwargs':
                            {
                                'fields_group': 'feature'
                            }
                    }],
                'learn_processors': [
                    {
                        'class': 'DropnaLabel'
                    },
                    {
                        'class': 'CSRankNorm',
                        'kwargs':
                            {
                                'fields_group': 'label'
                            }
                    }],
                'label': ['Ref($close, -1) / $close - 1']
            }
    }
    # 定义数据处理器参数
    segments = {'train': (args.train_start_date, args.train_end_date),
                'valid': (args.valid_start_date, args.valid_end_date),
                'test': (args.test_start_date, args.test_end_date)}  # 定义数据集划分
    # Qlib中创建数据集的方式
    dataset = DatasetH(hanlder, segments)  # 创建数据集

    df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"], col_set=["feature", "label"],
                                                  data_key=DataHandlerLP.DK_L, )  # 准备数据集

    df_train.to_csv("df_train.csv")  # 将训练集保存为 CSV 文件
    df_valid.to_csv("df_valid.csv")  # 将验证集保存为 CSV 文件
    df_test.to_csv("df_test.csv")  # 将测试集保存为 CSV 文件

    import pickle5 as pickle
    with open(args.market_value_path, "rb") as fh:
        df_market_value = pickle.load(fh)  # 读取市值数据
    df_market_value = df_market_value / 1000000000  # 对市值进行归一化
    stock_index = np.load(args.stock_index, allow_pickle=True).item()  # 加载股票索引数据

    start_index = 0  # 设置起始索引
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))  # 定义切片
    df_train['market_value'] = df_market_value[slc]  # 设置训练集市值数据
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())  # 缺失值填充为均值
    df_train['stock_index'] = 733  # 设置股票索引
    # 处理股票索引，stock_index应该是一个map，stock_index中与instrument索引中匹配的赋值到新列stock_index中，不匹配的默认为733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(
        int)

    print(df_train.shape)
    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['market_value'], df_train['stock_index'],
                              batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index,
                              device=device)  # 创建训练数据加载器

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))  # 定义切片
    df_valid['market_value'] = df_market_value[slc]  # 设置验证集市值数据
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())  # 缺失值填充为均值
    df_valid['stock_index'] = 733  # 设置股票索引
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(
        int)  # 处理股票索引
    start_index += len(df_valid.groupby(level=0).size())  # 更新起始索引

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['market_value'], df_valid['stock_index'],
                              pin_memory=True, start_index=start_index, device=device)  # 创建验证数据加载器

    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))  # 定义切片
    df_test['market_value'] = df_market_value[slc]  # 设置测试集市值数据
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())  # 缺失值填充为均值
    df_test['stock_index'] = 733  # 设置股票索引
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(
        int)  # 处理股票索引
    start_index += len(df_test.groupby(level=0).size())  # 更新起始索引

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'],
                             pin_memory=True, start_index=start_index, device=device)  # 创建测试数据加载器

    return train_loader, valid_loader, test_loader  # 返回训练、验证和测试数据加载器


# 主函数
def main(args):
    provider_uri = "/root/autodl-tmp/qlib/qlib_data/cn_data"  # 设置数据提供路径(autodl)
    # provider_uri = "/home/yujunpeng/autodl-tmp/qlib/qlib_data/cn_data"  # 设置数据提供路径(server3)
    qlib.qlib.init(provider_uri=provider_uri, region=REG_CN)  # 初始化 Qlib
    print('******************************')
    # seed = np.random.randint(1000000)
    seed = args.seed    # 随机种子
    random.seed(seed)
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 cpu 随机种子
    torch.cuda.manual_seed(seed)  # 设置 gpu 随机种子
    # suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s" % (  # 定义模型参数后缀
    #     args.model_name, args.hidden_size, args.num_layers, args.dropout,
    #     args.lr, args.batch_size, args.seed, args.annot
    # )

    output_path = args.outdir  # 输出路径
    # if not output_path:  # 如果未提供输出路径
    #     output_path = './output/' + suffix  # 设置默认输出路径
    # if not os.path.exists(output_path):  # 如果输出路径不存在
    #     os.makedirs(output_path)  # 创建输出路径

    # if not args.overwrite and os.path.exists(output_path + '/' + 'info.json'):  # 如果不覆盖且已存在信息文件
    #     print('already runned, exit.')  # 提示已运行
    #     return  # 退出

    # writer = SummaryWriter(log_dir=output_path)  # 创建 TensorBoard 日志记录器

    global global_log_file  # 全局日志文件路径
    global_log_file = output_path + '/' + args.name + '_run.log'  # 设置全局日志文件路径

    pprint('create loaders...')  # 输出日志
    train_loader, valid_loader, test_loader = create_loaders(args)  # 创建数据加载器

    stock2concept_matrix = np.load(args.stock2concept_matrix)  # 加载股票与概念关联矩阵
    if args.model_name == 'HIST':  # 如果模型为 HIST
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)  # 转换为张量并移到指定设备


    best_params = ""
    all_best_score = -np.inf
    res = []
    for alpha in [2]:
        for beta in [0.7]:
            for k in [7, 9]:
    # for alpha in (1.5, 2, 2.5):
    #     for beta in (0.6, 0.65, 0.7, 0.75, 0.8):
    #         for k in (5, 7, 9, 11, 13):

                suffix = "%s_alpha%s_beta%s_k%s" % (  # 定义模型参数后缀
                    "Att_" + args.model_name, alpha, beta, k
                )
                pprint(suffix)
                # 创建一个SummaryWriter实例（日志将被保存在"./logs"目录下）
                writer = SummaryWriter('runs/model_visualization/' + suffix)


                pprint('create model...')  # 输出日志
                model = get_model(args.model_name)(nnodes=args.nnodes, embedding_size=args.embedding_size,beta=beta,k=k,alpha=alpha,d_feat=args.d_feat, hidden_size=args.hidden_size, num_layers=args.num_layers)
                model.to(device)  # 将模型移到指定设备

                # # dummy_input = (torch.randn(128, 360).to(device), torch.randint(0, 1, (128, 20)).to(device))
                # dummy_input = (torch.randn(256, 360).to(device), torch.randint(0, 360, (256,)).to(device))
                # #
                # writer.add_graph(model,dummy_input)
                # writer.close()

                optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 使用 Adam 优化器
                best_score = -np.inf  # 初始化最佳分数
                best_epoch = 0  # 初始化最佳轮次
                stop_round = 0  # 初始化早停轮数
                best_param = copy.deepcopy(model.state_dict())  # 复制最佳参数
                params_list = collections.deque(maxlen=args.smooth_steps)  # 创建参数列表
                for epoch in range(args.n_epochs):  # 迭代训练轮次
                    pprint('Epoch:', epoch)  # 输出日志

                    pprint('training...')  # 输出日志
                    train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix)  # 训练模型


                    params_ckpt = copy.deepcopy(model.state_dict())  # 复制模型参数
                    params_list.append(params_ckpt)  # 将模型参数添加到列表
                    avg_params = average_params(params_list)  # 计算平均参数
                    model.load_state_dict(avg_params)  # 加载平均参数

                    pprint('evaluating...')  # 输出日志
                    train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model,
                                                                                                                 train_loader,
                                                                                                                 writer, args,
                                                                                                                 stock2concept_matrix,
                                                                                                                 prefix='Train')  # 计算训练集评价指标
                    val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader,
                                                                                                     writer, args,
                                                                                                     stock2concept_matrix,
                                                                                                     prefix='Valid')  # 计算验证集评价指标
                    test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model,
                                                                                                           test_loader, writer,
                                                                                                           args,
                                                                                                           stock2concept_matrix,
                                                                                                           prefix='Test')  # 计算测试集评价指标

                    pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss, val_loss, test_loss))  # 输出日志

                    pprint('train_score %.6f, valid_score %.6f, test_score %.6f' % (train_score, val_score, test_score))  # 输出日志
                    pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, val_ic, test_ic))  # 输出日志
                    pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (
                        train_rank_ic, val_rank_ic, test_rank_ic))  # 输出日志
                    pprint('Train Precision: ', train_precision)  # 输出日志
                    pprint('Valid Precision: ', val_precision)  # 输出日志
                    pprint('Test Precision: ', test_precision)  # 输出日志
                    pprint('Train Recall: ', train_recall)  # 输出日志
                    pprint('Valid Recall: ', val_recall)  # 输出日志
                    pprint('Test Recall: ', test_recall)  # 输出日志


                    if val_score > best_score:  # 如果验证分数更好

                        best_score = val_score  # 更新最佳分数

                        best_epoch = epoch  # 更新最佳轮次
                        best_loss = test_loss
                        best_precision = test_precision
                        best_recall = test_recall
                        best_ic = test_ic
                        best_rank_ic = test_rank_ic

                        best_param = copy.deepcopy(model.state_dict())  # 复制最佳参数
                        stop_round = 0  # 重置早停轮数
                    else:  # 否则
                        stop_round += 1  # 增加早停轮数
                        if stop_round >= args.early_stop:  # 如果早停轮数达到设定值
                            pprint('Early stopping at', epoch, 'with best epoch:', best_epoch)  # 输出日志
                            break  # 退出循环
                    model.load_state_dict(params_ckpt)  # 加载模型参数
                # model.load_state_dict(best_param)  # 加载最佳参数
                # train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model,
                #                                                                                              train_loader,
                #                                                                                              writer, args,
                #                                                                                              stock2concept_matrix,
                #                                                                                              prefix='Train')  # 计算训练集评价指标
                # val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader,
                #                                                                                  writer, args,
                #                                                                                  stock2concept_matrix,
                #                                                                                  prefix='Valid')  # 计算验证集评价指标
                # test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model,
                #                                                                                        test_loader, writer,
                #                                                                                        args,
                #                                                                                        stock2concept_matrix,
                #                                                                                        prefix='Test')  # 计算测试集评价指标

                res_sub = suffix + "_IC:" + str(best_ic) + "_RankIC:" + str(best_rank_ic)
                res.append(res_sub)
                pprint('best_loss:', best_loss)
                pprint('best score:', best_ic)
                pprint(' best_precision:', best_precision)
                pprint(' best_recall:', best_recall)
                pprint(' best_ic:', best_ic)
                pprint(' best_rank_ic:', best_rank_ic)
                pprint('best_epoch', best_epoch)

                if best_ic > all_best_score:
                    all_best_score = best_ic
                    best_params = suffix
                if not os.path.exists(output_path + '/' + suffix):
                    os.mkdir(output_path + '/' + suffix)
                torch.save(best_param, output_path + '/' + suffix + '/model.bin.e')  # 保存模型参数
    pprint("res")
    pprint(res)
    pprint("best_params")
    pprint(best_params)
    pprint("all_best_score")
    pprint(all_best_score)

def main2(args):
    provider_uri = "/root/autodl-tmp/qlib/qlib_data/cn_data"  # 设置数据提供路径(autodl)
    # provider_uri = "/home/yujunpeng/autodl-tmp/qlib/qlib_data/cn_data"  # 设置数据提供路径(server3)
    qlib.qlib.init(provider_uri=provider_uri, region=REG_CN)  # 初始化 Qlib
    print('******************************')
    # seed = np.random.randint(1000000)
    seed = args.seed    # 随机种子
    random.seed(seed)
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 cpu 随机种子
    torch.cuda.manual_seed(seed)  # 设置 gpu 随机种子
    # suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s" % (  # 定义模型参数后缀
    #     args.model_name, args.hidden_size, args.num_layers, args.dropout,
    #     args.lr, args.batch_size, args.seed, args.annot
    # )

    output_path = args.outdir  # 输出路径
    # if not output_path:  # 如果未提供输出路径
    #     output_path = './output/' + suffix  # 设置默认输出路径
    # if not os.path.exists(output_path):  # 如果输出路径不存在
    #     os.makedirs(output_path)  # 创建输出路径

    # if not args.overwrite and os.path.exists(output_path + '/' + 'info.json'):  # 如果不覆盖且已存在信息文件
    #     print('already runned, exit.')  # 提示已运行
    #     return  # 退出

    # writer = SummaryWriter(log_dir=output_path)  # 创建 TensorBoard 日志记录器

    global global_log_file  # 全局日志文件路径
    global_log_file = output_path + '/' + args.name + '_run.log'  # 设置全局日志文件路径

    pprint('create loaders...')  # 输出日志
    train_loader, valid_loader, test_loader = create_loaders(args)  # 创建数据加载器

    exit(0)

    stock2concept_matrix = np.load(args.stock2concept_matrix)  # 加载股票与概念关联矩阵
    if args.model_name == 'HIST':  # 如果模型为 HIST
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)  # 转换为张量并移到指定设备

    # 创建一个SummaryWriter实例（日志将被保存在"./logs"目录下）
    writer = SummaryWriter('runs/model_visualization/')


    pprint('create model...')  # 输出日志
    model = get_model(args.model_name)(nnodes=args.nnodes, embedding_size=args.embedding_size,beta=args.beta,k=args.k,alpha=args.alpha,d_feat=args.d_feat, hidden_size=args.hidden_size, num_layers=args.num_layers)
    model.to(device)  # 将模型移到指定设备

    # # dummy_input = (torch.randn(128, 360).to(device), torch.randint(0, 1, (128, 20)).to(device))
    # dummy_input = (torch.randn(256, 360).to(device), torch.randint(0, 360, (256,)).to(device))
    # #
    # writer.add_graph(model,dummy_input)
    # writer.close()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 使用 Adam 优化器
    best_score = -np.inf  # 初始化最佳分数
    best_epoch = 0  # 初始化最佳轮次
    stop_round = 0  # 初始化早停轮数
    best_param = copy.deepcopy(model.state_dict())  # 复制最佳参数
    params_list = collections.deque(maxlen=args.smooth_steps)  # 创建参数列表
    for epoch in range(args.n_epochs):  # 迭代训练轮次
        pprint('Epoch:', epoch)  # 输出日志

        pprint('training...')  # 输出日志
        train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix)  # 训练模型


        params_ckpt = copy.deepcopy(model.state_dict())  # 复制模型参数
        params_list.append(params_ckpt)  # 将模型参数添加到列表
        avg_params = average_params(params_list)  # 计算平均参数
        model.load_state_dict(avg_params)  # 加载平均参数

        pprint('evaluating...')  # 输出日志
        train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model,
                                                                                                     train_loader,
                                                                                                     writer, args,
                                                                                                     stock2concept_matrix,
                                                                                                     prefix='Train')  # 计算训练集评价指标
        val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader,
                                                                                         writer, args,
                                                                                         stock2concept_matrix,
                                                                                         prefix='Valid')  # 计算验证集评价指标
        test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model,
                                                                                               test_loader, writer,
                                                                                               args,
                                                                                               stock2concept_matrix,
                                                                                               prefix='Test')  # 计算测试集评价指标

        pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss, val_loss, test_loss))  # 输出日志

        pprint('train_score %.6f, valid_score %.6f, test_score %.6f' % (train_score, val_score, test_score))  # 输出日志
        pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, val_ic, test_ic))  # 输出日志
        pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (
            train_rank_ic, val_rank_ic, test_rank_ic))  # 输出日志
        pprint('Train Precision: ', train_precision)  # 输出日志
        pprint('Valid Precision: ', val_precision)  # 输出日志
        pprint('Test Precision: ', test_precision)  # 输出日志
        pprint('Train Recall: ', train_recall)  # 输出日志
        pprint('Valid Recall: ', val_recall)  # 输出日志
        pprint('Test Recall: ', test_recall)  # 输出日志


        if val_score > best_score:  # 如果验证分数更好

            best_score = val_score  # 更新最佳分数

            best_epoch = epoch  # 更新最佳轮次

            best_loss = test_loss
            best_precision = test_precision
            best_recall = test_recall
            best_ic = test_ic
            best_rank_ic = test_rank_ic

            best_param = copy.deepcopy(model.state_dict())  # 复制最佳参数
            stop_round = 0  # 重置早停轮数
        else:  # 否则
            stop_round += 1  # 增加早停轮数
            if stop_round >= args.early_stop:  # 如果早停轮数达到设定值
                pprint('Early stopping at', epoch, 'with best epoch:', best_epoch)  # 输出日志
                break  # 退出循环
        model.load_state_dict(params_ckpt)  # 加载模型参数
    # model.load_state_dict(best_param)  # 加载最佳参数
    # train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model,
    #                                                                                              train_loader,
    #                                                                                              writer, args,
    #                                                                                              stock2concept_matrix,
    #                                                                                              prefix='Train')  # 计算训练集评价指标
    # val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader,
    #                                                                                  writer, args,
    #                                                                                  stock2concept_matrix,
    #                                                                                  prefix='Valid')  # 计算验证集评价指标
    # test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model,
    #                                                                                        test_loader, writer,
    #                                                                                        args,
    #                                                                                        stock2concept_matrix,
    #                                                                                        prefix='Test')  # 计算测试集评价指标

    # best_loss = test_loss
    # best_precision = test_precision
    # best_recall = test_recall
    # best_ic = test_ic
    # best_rank_ic = test_rank_ic
    pprint('best_loss:', best_loss)
    pprint('best score:', best_ic)
    pprint(' best_precision:', best_precision)
    pprint(' best_recall:', best_recall)
    pprint(' best_ic:', best_ic)
    pprint(' best_rank_ic:', best_rank_ic)
    pprint('best_epoch', best_epoch)
    pprint("args_alpha", args.alpha)
    pprint("args_beta", args.beta)
    pprint("args_k", args.k)


    torch.save(best_param, output_path + '/model.bin.e')  # 保存模型参数




def test(args):

    provider_uri = "/root/autodl-tmp/qlib/qlib_data/cn_data"  # 设置数据提供路径(autodl)
    # provider_uri = "/home/yujunpeng/autodl-tmp/qlib/qlib_data/cn_data"  # 设置数据提供路径(server3)
    qlib.qlib.init(provider_uri=provider_uri, region=REG_CN)  # 初始化 Qlib
    print('******************************')
    # seed = np.random.randint(1000000)
    seed = args.seed    # 随机种子
    random.seed(seed)
    np.random.seed(seed)  # 设置 NumPy 随机种子
    torch.manual_seed(seed)  # 设置 cpu 随机种子
    torch.cuda.manual_seed(seed)  # 设置 gpu 随机种子
    # suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s" % (  # 定义模型参数后缀
    #     args.model_name, args.hidden_size, args.num_layers, args.dropout,
    #     args.lr, args.batch_size, args.seed, args.annot
    # )

    output_path = args.outdir  # 输出路径

    global global_log_file  # 全局日志文件路径
    global_log_file = output_path + '/' + args.name + '_run.log'  # 设置全局日志文件路径

    pprint('create loaders...')  # 输出日志
    train_loader, valid_loader, test_loader = create_loaders(args)  # 创建数据加载器

    stock2concept_matrix = np.load(args.stock2concept_matrix)  # 加载股票与概念关联矩阵
    if args.model_name == 'HIST':  # 如果模型为 HIST
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)  # 转换为张量并移到指定设备

    # 创建一个SummaryWriter实例（日志将被保存在"./logs"目录下）
    writer = SummaryWriter('runs/model_visualization/')


    pprint('create model...')  # 输出日志
    model = get_model(args.model_name)(nnodes=args.nnodes, embedding_size=args.embedding_size,beta=args.beta,k=args.k,alpha=args.alpha,d_feat=args.d_feat, hidden_size=args.hidden_size, num_layers=args.num_layers)
    model.to(device)  # 将模型移到指定设备
    avg_params = torch.load("model.bin.e")
    model.load_state_dict(avg_params)  # 加载平均参数
    val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(1, model, valid_loader,
                                                                                     writer, args,
                                                                                     stock2concept_matrix,
                                                                                     prefix='Valid')  # 计算验证集评价指标
    test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(1, model,
                                                                                           test_loader, writer,
                                                                                           args,
                                                                                           stock2concept_matrix,
                                                                                           prefix='Test')  # 计算测试集评价指标

    pprint('valid_ic %.6f, test_ic %.6f' % (val_ic, test_ic))  # 输出日志
    pprint('valid_rank_ic %.6f, test_rank_ic %.6f' % (
        val_rank_ic, test_rank_ic))  # 输出日志

def set_seed(args):
    seed = args.seed
    # 下面两个常规设置了，用来np和random的话要设置
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个

    # torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。


class ParseConfigFile(argparse.Action):  # 解析配置文件类

    def __call__(self, parser, namespace, filename, option_string=None):  # 调用函数

        if not os.path.exists(filename):  # 如果文件不存在
            raise ValueError('cannot find config at `%s`' % filename)  # 抛出异常

        with open(filename) as f:  # 打开文件
            config = json.load(f)  # 加载配置
            for key, value in config.items():  # 遍历配置项
                setattr(namespace, key, value)  # 设置属性值


def parse_args():  # 解析命令行参数函数

    parser = argparse.ArgumentParser()  # 创建参数解析器

    # 模型参数
    parser.add_argument('--model_name', default='ALSGCN')  # 模型名称，默认为 ALSGCN
    parser.add_argument('--nnodes', default=735)  # 股票总数量，735
    parser.add_argument('--embedding_size', default=40) # embedding的大小，默认40
    parser.add_argument('--beta', default=0.8) # beta,默认0.5
    parser.add_argument('--k', default=5)  # beta,默认20
    parser.add_argument('--alpha', default=2) # alpha, 默认3 csi100:2最好 csi300:2.5最好
    parser.add_argument('--mu', default=0.3) # mu, 默认0.3

    parser.add_argument('--d_feat', type=int, default=6)  # 特征维度，默认为6
    parser.add_argument('--hidden_size', type=int, default=128)  # 隐藏层大小，默认为128
    parser.add_argument('--num_layers', type=int, default=2)  # 层数，默认为2
    parser.add_argument('--dropout', type=float, default=0.0)  # Dropout概率，默认为0.0
    parser.add_argument('--K', type=int, default=1)  # K值，默认为1

    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=200)  # 训练轮次，默认为200
    parser.add_argument('--lr', type=float, default=2e-4)  # 学习率，默认为2e-4
    parser.add_argument('--early_stop', type=int, default=15)  # 早停轮次，默认为30
    parser.add_argument('--smooth_steps', type=int, default=5)  # 平滑步数，默认为5
    parser.add_argument('--metric', default='IC')  # 评价指标，默认为IC
    # parser.add_argument('--repeat', type=int, default=10)  # 重复次数，默认为10

    # 数据参数
    parser.add_argument('--data_set', type=str, default='csi100')  # 数据集名称，默认为csi100
    parser.add_argument('--pin_memory', action='store_false', default=True)  # 是否将数据存储在固定内存中，默认为True
    parser.add_argument('--batch_size', type=int, default=-1)  # 批大小，默认为-1表示每日批处理
    parser.add_argument('--least_samples_num', type=float, default=1137.0)  # 最小样本数，默认为1137.0
    parser.add_argument('--label', default='')  # 指定其他标签，默认为空
    parser.add_argument('--train_start_date', default='2007-01-01')  # 训练集起始日期，默认为2007-01-01
    parser.add_argument('--train_end_date', default='2014-12-31')  # 训练集结束日期，默认为2014-12-31
    parser.add_argument('--valid_start_date', default='2015-01-01')  # 验证集起始日期，默认为2015-01-01
    parser.add_argument('--valid_end_date', default='2016-12-31')  # 验证集结束日期，默认为2016-12-31
    parser.add_argument('--test_start_date', default='2017-01-01')  # 测试集起始日期，默认为2017-01-01
    parser.add_argument('--test_end_date', default='2020-12-31')  # 测试集结束日期，默认为2020-12-31

    # 其他参数
    parser.add_argument('--seed', type=int, default=0)  # 随机种子，默认为0
    parser.add_argument('--annot', default='')  # 注释，默认为空
    parser.add_argument('--config', action=ParseConfigFile, default='')  # 配置文件，默认为空
    parser.add_argument('--name', type=str, default='csi100_HIST')  # 名称，默认为csi100_HIST

    # CSI 300 输入参数
    parser.add_argument('--market_value_path',
                        default='./data/csi300_market_value_07to20.pkl')  # 市值路径，默认为'./data/csi300_market_value_07to20.pkl'
    parser.add_argument('--stock2concept_matrix',
                        default='./data/csi300_stock2concept.npy')  # 股票与概念矩阵路径，默认为'./data/csi300_stock2concept.npy'
    parser.add_argument('--stock_index',
                        default='./data/csi300_stock_index.npy')  # 股票索引路径，默认为'./data/csi300_stock_index.npy'

    parser.add_argument('--outdir', default='./output/csi100_HIST')  # 输出目录，默认为'./output/csi100_HIST'
    parser.add_argument('--overwrite', action='store_true', default=False)  # 是否覆盖，默认为False

    args = parser.parse_args()  # 解析参数

    return args  # 返回参数


if __name__ == '__main__':  # 如果运行为主程序
    # provider_uri = "~/.qlib/qlib_data/cn_data"  # 设置数据提供路径
    print('******************************')

    args = parse_args()  # 解析命令行参数
    main2(args)  # 调用主函数进行训练与评估
