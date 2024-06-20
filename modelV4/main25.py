#加时间周期性

import pandas as pd
import numpy as np
import torch
from stid.stid_arch import STID
from util import *
#设置读取前2880行数据包括header
data=pd.read_csv('.\data_all_time_linear_time.csv')
df=data[data.columns[1:]]
data2=pd.read_csv(r'.\20231205_data_all_time_linear_time.csv')
data2=data2[data2.columns[1:]]
# df=df.clip(upper=60)
# data2=data2.clip(upper=60)
series_len=20
del data
# 选取这个时间范围是因为目前模型过拟合了，就是train很低但是valid的很高，test也很高
# start=4000
# stop=7000
# start=5000
# stop=9000
start=2000
stop=5000

df1=data2[start+series_len:series_len+stop]
del data2

df2=df[start:stop+series_len+1000]

df1_values=np.expand_dims(df1.values, axis=-1)
df2_values=np.expand_dims(df2.values, axis=-1)

mask=np.load('.\mask_before.npy')
#mask前1440行，即前一天的mask
#mask后1440行，即后一天的mask
mask2=mask[start+series_len:stop+series_len]
mask2_values=np.expand_dims(mask2, axis=-1)


target_len=10
seq_len=20
num_features=15
num_steps=1
step_length=17

expand_values=df1_values
mask=mask2_values

# expand_values_m=expand_values
# x,_=sliding_windows(expand_values_m,seq_len*num_steps,num_features,target_len=target_len,step_length=step_length)
x,y=sliding_windows(expand_values,seq_len*num_steps,num_features,target_len=target_len,step_length=step_length)
x_mask,y_mask=sliding_windows(mask,seq_len*num_steps,num_features,target_len=target_len,step_length=step_length)
# x_mask2,y_mask2=sliding_windows(mask,seq_len*num_steps,num_features,target_len=target_len,step_length=step_length)
# x_m,y_m=sliding_windows(m,seq_len*num_steps,num_features,target_len=target_len)
x=x[:,::num_steps]
x_mask=x_mask[:,::num_steps]


x2,_=sliding_windows(df2_values,seq_len*num_steps+series_len*2+num_features+target_len,0,step_length=step_length)
x2=x2[::num_steps]
x2=x2[:len(x)]

# x=x*x_mask
# x2=
# del m
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device=torch.device("cpu")

train_size = int(len(y) * 0.6)
valid_size = int(len(y) * 0.2)
test_size = len(y) - train_size - valid_size
#随机打乱数据
# index = np.arange(len(y))
# np.random.shuffle(index)
# x=x[index]
# y=y[index]
# x2=x2[index]
# x_mask=x_mask[index]
# y_mask=y_mask[index]
# # x_m=x_m[index]
# x_mask=np.ones_like(x_mask)
# y_mask=np.ones_like(y_mask)
# x=np.einsum('ijkm,ijkm->ijkm',x,x_mask)

# x=x*x_mask
train_x = torch.from_numpy(x[:train_size]).type(torch.Tensor).to(device)
mask_train_x = torch.from_numpy(x_mask[:train_size]).type(torch.Tensor).to(device)
x_train_data=train_x
train_x2 = torch.from_numpy(x2[:train_size]).type(torch.Tensor).to(device)

train_y = torch.from_numpy(y[:train_size]).type(torch.Tensor).to(device)
mask_train_y = torch.from_numpy(y_mask[:train_size]).type(torch.Tensor).to(device)
y_train_data=train_y

valid_x = torch.from_numpy(x[train_size:train_size+valid_size]).type(torch.Tensor).to(device)
mask_valid_x = torch.from_numpy(x_mask[train_size:train_size+valid_size]).type(torch.Tensor).to(device)
x_valid_data=valid_x
valid_x2 = torch.from_numpy(x2[train_size:train_size+valid_size]).type(torch.Tensor).to(device)


valid_y = torch.from_numpy(y[train_size:train_size+valid_size]).type(torch.Tensor).to(device)
mask_valid_y = torch.from_numpy(y_mask[train_size:train_size+valid_size]).type(torch.Tensor).to(device)
y_valid_data=valid_y

test_x = torch.from_numpy(x[train_size+valid_size:]).type(torch.Tensor).to(device)
mask_test_x = torch.from_numpy(x_mask[train_size+valid_size:]).type(torch.Tensor).to(device)
x_test_data=test_x
test_x2 = torch.from_numpy(x2[train_size+valid_size:]).type(torch.Tensor).to(device)

test_y = torch.from_numpy(y[train_size+valid_size:]).type(torch.Tensor).to(device)
mask_test_y = torch.from_numpy(y_mask[train_size+valid_size:]).type(torch.Tensor).to(device)
y_test_data=test_y


del x ,y ,x_mask ,y_mask ,train_x ,train_y ,valid_x ,valid_y ,test_x ,test_y 
adj=np.load('./adj.npy')

# D = np.diag(np.sum(adj, axis=1))

# # 计算标准化的邻接矩阵
# adj = np.linalg.inv(D) @ adj
# adj=torch.from_numpy(adj).type(torch.LongTensor).to(device)
adj=torch.from_numpy(adj).type(torch.Tensor).to(device)
parameter= {
    "num_nodes": 7567,
    "input_len": seq_len,
    "embed_dim": 64,
    "output_len": num_features,
    "num_layer": 2,
    "if_node":True ,
    "node_dim": 32,
    "if_T_i_D": True,
    "if_D_i_W": False,
    "if_time_series": True,
    "temp_dim_tid": 64,
    "temp_dim_diw": 32,
    "time_of_day_size": 100,
    "day_of_week_size": 7,
    "is_Smooth":False,
    "gcn_bool": True,
    "out_gcn_dim": 32,
    "t_i_d_len":seq_len*num_steps+series_len*2+num_features+target_len
}

model=STID(adj=adj,**parameter).to(device)
lr=0.005
weight_decay=0.001
criterion=mae_loss
# criterion=masked_mape_loss
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# batch_size=len(x_train_data)
batch_size=64
model.train()
for epoch in range(1,151):
    loss_data=[]
    rmse_data=[]
    mape_data=[]
    loss_data_9_10=[]
    rmse_data_9_10=[]
    mape_data_9_10=[]
    for i in range(0,len(x_train_data),batch_size):
        x_train_data_batch=x_train_data[i:i+batch_size]
        y_train_data_batch=y_train_data[i:i+batch_size]
        mask_train_y_batch=mask_train_y[i:i+batch_size]
        outputs = model(x_train_data_batch,train_x2[i:i+batch_size])
        # outputs=exponential_smoothing(outputs, alpha=0.9)
        # outputs=torch.exp(outputs/2)
        # outputs=outputs[:,9:10,:,:]
        # outputs=torch.mean(outputs, dim=1,keepdim=True)
        loss = criterion(outputs, y_train_data_batch,mask_train_y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_data.append(loss.item())
        rmse_data.append(masked_rmse_loss(outputs, y_train_data_batch,mask_train_y_batch).item())
        mape_data.append(masked_mape_loss(outputs, y_train_data_batch,mask_train_y_batch).item())
        loss_data_9_10.append(criterion(outputs[:,9:10,:,:], y_train_data_batch[:,9:10,:,:],mask_train_y_batch[:,9:10,:,:]).item())
        rmse_data_9_10.append(masked_rmse_loss(outputs[:,9:10,:,:], y_train_data_batch[:,9:10,:,:],mask_train_y_batch[:,9:10,:,:]).item())
        mape_data_9_10.append(masked_mape_loss(outputs[:,9:10,:,:], y_train_data_batch[:,9:10,:,:],mask_train_y_batch[:,9:10,:,:]).item())
    print("Epoch: %d, loss: %1.5f" % (epoch, np.mean(np.array(loss_data))))
    print("Epoch: %d, rmse loss: %1.5f" % (epoch, np.mean(np.array(rmse_data))))
    print("Epoch: %d, mape loss: %1.5f" % (epoch, np.mean(np.array(mape_data))))
    print("Epoch: %d, loss_9_10: %1.5f" % (epoch, np.mean(np.array(loss_data_9_10))))
    print("Epoch: %d, rmse loss_9_10: %1.5f" % (epoch, np.mean(np.array(rmse_data_9_10))))
    print("Epoch: %d, mape loss_9_10: %1.5f" % (epoch, np.mean(np.array(mape_data_9_10))))
    if epoch % 3 == 0:
        model.eval()
        valid_outputs = model(x_valid_data,valid_x2)
        valid_loss = criterion(valid_outputs, y_valid_data,mask_valid_y)
        print("Epoch: %d, valid_loss: %1.5f" % (epoch, valid_loss.item()))
        print("Epoch: %d, valid_rmse loss: %1.5f" % (epoch, masked_rmse_loss(valid_outputs, y_valid_data,mask_valid_y).item()))
        print("Epoch: %d, valid_mape loss: %1.5f" % (epoch, masked_mape_loss(valid_outputs, y_valid_data,mask_valid_y).item()))
        print("Epoch: %d, valid_loss_9_10: %1.5f" % (epoch, criterion(valid_outputs[:,9:10,:,:], y_valid_data[:,9:10,:,:],mask_valid_y[:,9:10,:,:]).item()))
        print("Epoch: %d, valid_rmse loss_9_10: %1.5f" % (epoch, masked_rmse_loss(valid_outputs[:,9:10,:,:], y_valid_data[:,9:10,:,:],mask_valid_y[:,9:10,:,:]).item()))
        print("Epoch: %d, valid_mape loss_9_10: %1.5f" % (epoch, masked_mape_loss(valid_outputs[:,9:10,:,:], y_valid_data[:,9:10,:,:],mask_valid_y[:,9:10,:,:]).item()))
        model.train()
   
# import sys
# sys.exit(0)
        
model.eval()
with torch.no_grad():
    test_outputs=model(x_test_data,test_x2)
    test_loss = criterion(test_outputs, y_test_data,mask_test_y)
    print("Epoch: %d, test_loss: %1.5f" % (epoch, test_loss.item()))
    print("Epoch: %d, test_rmse loss: %1.5f" % (epoch, masked_rmse_loss(test_outputs, y_test_data,mask_test_y).item()))
    print("Epoch: %d, test_mape loss: %1.5f" % (epoch, masked_mape_loss(test_outputs, y_test_data,mask_test_y).item()))
    # 分别验证0到1，以此类推的数据损失
    for i in range(num_features):
        print("Epoch: %d, test_loss_%d: %1.5f" % (epoch,i, criterion(test_outputs[:,i:i+1,:,:], y_test_data[:,i:i+1,:,:],mask_test_y[:,i:i+1,:,:]).item()))
        # print("Epoch: %d, test_rmse loss_%d: %1.5f" % (epoch,i, masked_rmse_loss(test_outputs[:,i:i+1,:,:], y_test_data[:,i:i+1,:,:],mask_test_y[:,i:i+1,:,:]).item()))
        print("Epoch: %d, test_mape loss_%d: %1.5f" % (epoch,i, masked_mape_loss(test_outputs[:,i:i+1,:,:], y_test_data[:,i:i+1,:,:],mask_test_y[:,i:i+1,:,:]).item()))
    # print("Epoch: %d, test_loss_9_10: %1.5f" % (epoch, criterion(test_outputs[:,9:10,:,:], y_test_data[:,9:10,:,:],mask_test_y[:,9:10,:,:]).item()))
    # print("Epoch: %d, test_rmse loss_9_10: %1.5f" % (epoch, masked_rmse_loss(test_outputs[:,9:10,:,:], y_test_data[:,9:10,:,:],mask_test_y[:,9:10,:,:]).item()))
    # print("Epoch: %d, test_mape loss_9_10: %1.5f" % (epoch, masked_mape_loss(test_outputs[:,9:10,:,:], y_test_data[:,9:10,:,:],mask_test_y[:,9:10,:,:]).item()))
    # np.save('./outputs.npy',test_outputs.cpu().numpy())
    
        