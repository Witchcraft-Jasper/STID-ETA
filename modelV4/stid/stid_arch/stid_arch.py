import torch
from torch import nn

from .mlp import MultiLayerPerceptron
from .gcn import *
import numpy as np
from .util import *
class STID(nn.Module):
    def __init__(self,adj=None, **model_args):
        super().__init__()
        # attributes
        self.input_len = model_args["input_len"]
        self.output_len = model_args["output_len"]
        self.num_nodes = model_args["num_nodes"]
        self.if_node=model_args["if_node"]
        self.node_dim = model_args["node_dim"]
        self.num_layer = model_args["num_layer"]
        self.embed_dim = model_args["embed_dim"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.time_of_day_size = model_args["time_of_day_size"]
        
        self.if_time_series=True
        self.is_Smooth=False
        self.gcn_bool=True
        if "if_time_series" in model_args:
            self.if_time_series=model_args["if_time_series"]
        if "is_Smooth" in model_args:
            self.is_Smooth=model_args["is_Smooth"]
        if "gcn_bool" in model_args:
            self.gcn_bool=model_args["gcn_bool"]
            self.out_gcn_dim=model_args["out_gcn_dim"]
        if "t_i_d_len" in model_args:
            self.t_i_d_len=model_args["t_i_d_len"]
        self.adj=adj
        
        self.hidden_dim = self.embed_dim * int(self.if_time_series) +self.node_dim * int(self.if_node)+self.temp_dim_tid * int(self.if_time_in_day)
        self.fc_dim=self.hidden_dim+self.out_gcn_dim*int(self.gcn_bool)*0
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        # self.node_emb = nn.Embedding()
        # nn.init.xavier_uniform_(self.node_emb)
        # if self.if_time_in_day:
        self.time_in_day_emb=nn.Sequential(
            nn.Conv2d(in_channels=self.t_i_d_len, out_channels=self.temp_dim_tid, kernel_size=(1, 1), bias=True),
            nn.Conv2d(in_channels=self.temp_dim_tid, out_channels=self.temp_dim_tid, kernel_size=(1, 1), bias=True),
            # nn.BatchNorm2d(self.temp_dim_tid),
            # nn.Dropout(0.2),
        )
            # self.time_in_day_emb= nn.Conv2d(in_channels=self.input_len, out_channels=self.temp_dim_tid, kernel_size=(1, 1), bias=True)
            # self.time_in_day_emb2= nn.Conv2d(in_channels=self.temp_dim_tid, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        # self.adj=adj
        # nn.init.xavier_uniform_(self.adj)
        # self.emdedding_node=nn.Embedding(self.num_nodes,self.node_dim)
        # self.emdedding_node=nn.Linear(self.num_nodes,self.node_dim)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
                # embedding layer
        if self.if_time_series:
            self.time_series_emb_layer = nn.Conv2d(
                 in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        # if self.gcn_bool:
        #     self.gcn=graphConvolution(self.input_len, self.out_gcn_dim)
            
        # self.dropout = nn.Dropout(0.1)
        self.batch_norm=nn.BatchNorm2d(self.hidden_dim,eps=0.001)
        self.regression_layer = nn.Conv2d(in_channels=self.fc_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        if self.is_Smooth:
            kernel_size = 20
            padding = kernel_size // 2
            self.smooth= nn.Conv2d(
                in_channels=self.output_len, out_channels=self.output_len,kernel_size=kernel_size,padding=padding, bias=False)
        
        # self.gcn_var=nn.parameter([0.2],requires_grad=True,dtype=torch.float32)
        # if(self.is_Smooth):
        #     self.poly=polyFit()
            # self.poly=nn.Conv2d(in_channels=self.output_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        # self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
# 整个模型存在两个问题，一个是seriesemb太大，一个是nodeemb太小
    def forward(self, history_data: torch.Tensor,history_data_2=None) -> torch.Tensor:

        # encoding 第一个维度是batch_size 第二个维度是seq_len 第三个维度是num_nodes 第四个维度是input_dim
        # prediction=nn.ReLU()(prediction)
        # history_data=torch.einsum('ijkl,mk->ijml',history_data,self.adj)
        # history_data=torch.einsum('ijkl,kj->ijkl',history_data,self.weight)
        if self.gcn_bool:
            history_data=torch.einsum('ijkl,mk->ijml',history_data,self.adj)
            history_data_2=torch.einsum('ijkl,mk->ijml',history_data_2,self.adj)
        embeddings=[]
        if self.if_time_series:
            time_series_emb = self.time_series_emb_layer(history_data[..., 0:1])
            # time_in_day_data = history_data[..., 1:2]
            embeddings.append(time_series_emb)
            # embeddings.append(time_series_emb+torch.sigmoid(out)*time_series_emb)
        if self.if_time_in_day:
            out=self.time_in_day_emb(history_data_2)
            embeddings.append(out)
            
        # if self.if_time_in_day:
        #     out=self.time_in_day_emb(history_data_2)
        #     out=F.relu(out)
        #     embeddings.append(out)
        batch_size, _, num_nodes, _ = history_data.shape
        if self.if_node:
            out=self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            embeddings.append(out)
        hidden = torch.cat(embeddings, dim=1)
        # encoding
        # hidden=nn.ReLU()(hidden)
        hidden=self.batch_norm(hidden)
        hidden = self.encoder(hidden)

        # regression
        # if self.gcn_bool:
            # out_adj=self.gcn(history_data[...,0:1],self.adj)
            # hidden=torch.cat([hidden,out_adj],dim=1)
        
        prediction = self.regression_layer(hidden)
        prediction=F.relu(prediction)
        # prediction=torch.einsum('ijkl,mk->ijml', prediction, self.adj)
        if self.is_Smooth:
            #数据平滑
            U, S, V = torch.svd(prediction)
            k = 10  # 保留的奇异值的数量
            S[:,:10] = 0
            S_diag = torch.diag_embed(S)        
            prediction=torch.matmul(U, torch.matmul(S_diag, V.transpose(-2, -1)))
            # outputs=exponential_smoothing_4d(outputs,0.9)
        #对prediction进行平滑处理
        # prediction=self.dropout(prediction)
        # prediction=torch.clip(prediction,max=60)
        return prediction
