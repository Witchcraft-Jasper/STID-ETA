import numpy as np
import torch

import torch

def exponential_smoothing_4d(data, alpha):
    # 获取数据的形状
    batch_size, time_length, num_nodes, num_features = data.size()

    # 创建一个与数据形状相同的张量来保存平滑后的数据
    smoothed_data = torch.zeros_like(data)

    # 对每个批次、节点和特征进行指数平滑
    for i in range(batch_size):
        for j in range(num_nodes):
            for k in range(num_features):
                smoothed_data[i, 0, j, k] = data[i, 0, j, k]
                for t in range(1, time_length):
                    smoothed_data[i, t, j, k] = alpha * data[i, t, j, k] + (1 - alpha) * smoothed_data[i, t-1, j, k]

    return  alpha * data

def exponential_smoothing(x, alpha):
    smoothed = torch.zeros_like(x)
    smoothed[0] = x[0]
    for i in range(1, len(x)):
        smoothed[i] = alpha * x[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

def smooth_ercheng(data):
    data = data.clone()
    new_data = data.clone()  # 创建一个新的变量来存储结果
    # 我们想要对第二维度进行平滑处理
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            for k in range(data.shape[3]):
                # 提取第二维度的数据
                y = data[i, :, j, k].reshape(-1, 1)

                # 创建x值，这里我们假设x是等间距的
                x = torch.linspace(-1, 1, data.shape[1]).reshape(-1, 1)

                # 构造设计矩阵A
                A = torch.cat([x**0, x**1, x**2], dim=1)

                # 使用最小二乘法进行拟合
                solution, residuals, rank, s = torch.linalg.lstsq(A, y)

                # 使用拟合的多项式替换原始数据
                new_data[i, :, j, k] = A @ solution.squeeze()  # 修改这一行
    return new_data  # 返回新的变量

def generate_m(data,rate=0.1):
    mask=np.ones_like(data)
    mask=np.random.choice([0,1],size=mask.shape,p=[rate,1-rate])
    return mask


def generate_m_colums(data, rate=0.1):
    mask = np.ones_like(data)
    num_cols = mask.shape[1]
    num_zero_cols = int(rate * num_cols)  # 计算需要设为0的列的数量
    zero_cols = np.random.choice(num_cols, num_zero_cols, replace=False)  # 随机选择一些列索引
    mask[:, zero_cols] = 0  # 将这些列设为0
    return mask

def sliding_windows(data, seq_length, num_targets,target_len=0):
    x = []
    y = []
    for i in range(len(data)-seq_length-num_targets -target_len):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length+target_len:i+seq_length+num_targets+target_len]
        x.append(_x)
        y.append(_y)

    return np.stack(x), np.stack(y)
def masked_mse_loss(pred, target, mask):
    # 计算均方误差
    mse_loss = (pred - target) ** 2
    
    # 使用掩码过滤数据
    mask=mask.type(torch.bool)
    masked_loss = torch.masked_select(mse_loss, mask)
    
    # 计算均值
    mean_loss = masked_loss.sum() / mask.sum()
    return mean_loss

def masked_mape_loss(pred, target, mask):
    # 计算绝对误差
    zero_mask = (target != 0)
    mask=mask.type(torch.bool)
    mask = mask & zero_mask
    mae_loss = torch.abs(pred - target)/target
    # 使用掩码过滤数据
    masked_loss = torch.masked_select(mae_loss, mask.type(torch.bool))
    
    # 计算均值
    mean_loss = masked_loss.sum() / mask.sum()
    # if mask.sum().item()==0:
    #     return 0
    return mean_loss

def masked_rmse_loss(pred, target, mask):
    # 计算均方根误差
    mse_loss = (pred - target) ** 2
    
    # 使用掩码过滤数据
    mask=mask.type(torch.bool)
    masked_loss = torch.masked_select(mse_loss, mask)
    
    # 计算均值
    mean_loss = torch.sqrt(masked_loss.sum() / mask.sum())
    return mean_loss

def getdate(length,data_numpy):
    data_numpy_list=[]
    for i in range(0,len(data_numpy)-length,length):
        item1=data_numpy[i:i+length]
        item2=data_numpy[i+length:i+length*2]
        data_numpy_list.append(np.concatenate((item1,item2),axis=-1))
    return np.concatenate(data_numpy_list,axis=0)

def mae_loss(pred, target, mask):
    
    # 计算绝对误差
    mae_loss = torch.abs(pred - target)
    
    # 使用掩码过滤数据
    mask=mask.type(torch.bool)
    masked_loss = torch.masked_select(mae_loss, mask)

    return masked_loss.sum() / mask.sum()