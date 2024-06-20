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
        self.input_dim = model_args["input_dim"]
        self.if_node=model_args["if_node"]
        self.node_dim = model_args["node_dim"]
        self.num_layer = model_args["num_layer"]
        self.embed_dim = model_args["embed_dim"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.is_Smooth=False
        self.gcn_bool=True
        if "is_Smooth" in model_args:
            self.is_Smooth=model_args["is_Smooth"]
        if "gcn_bool" in model_args:
            self.gcn_bool=model_args["gcn_bool"]
            self.out_gcn_dim=model_args["out_gcn_dim"]
        self.adj=adj
        
        self.hidden_dim = self.embed_dim+self.node_dim * int(self.if_node)
        self.fc_dim=self.hidden_dim+self.out_gcn_dim*int(self.gcn_bool)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            
            self.time_in_day_emb=nn.Sequential(
                nn.Conv2d(in_channels=self.input_len, out_channels=self.temp_dim_tid, kernel_size=(1, 1), bias=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.temp_dim_tid, out_channels=self.temp_dim_tid, kernel_size=(1, 1), bias=True),
            )
            # self.time_in_day_emb= nn.Conv2d(in_channels=self.input_len, out_channels=self.temp_dim_tid, kernel_size=(1, 1), bias=True)
            # self.time_in_day_emb2= nn.Conv2d(in_channels=self.temp_dim_tid, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        # self.adj=adj
        # nn.init.xavier_uniform_(self.adj)
        self.emdedding_node=nn.Embedding(self.num_nodes,self.node_dim)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
                # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        if self.gcn_bool:
            self.gcn=graphConvolution(self.input_len, self.out_gcn_dim)
        
        
        
        # self.weight=nn.Parameter(torch.randn(self.num_nodes,self.input_len))
        # self.weight2=nn.Parameter(torch.randn(self.num_nodes,self.output_len))
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.xavier_uniform_(self.weight2)
        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.fc_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        # if(self.is_Smooth):
        #     self.poly=polyFit()
            # self.poly=nn.Conv2d(in_channels=self.output_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        # self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:

        # encoding 第一个维度是batch_size 第二个维度是seq_len 第三个维度是num_nodes 第四个维度是input_dim
        # prediction=nn.ReLU()(prediction)
        # history_data=torch.einsum('ijkl,mk->ijml',history_data,self.adj)
        # history_data=torch.einsum('ijkl,kj->ijkl',history_data,self.weight)
        time_series_emb = self.time_series_emb_layer(history_data[..., 0:1])
        embeddings = [time_series_emb]
        batch_size, _, num_nodes, _ = history_data.shape
        if self.if_node:
            out=self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            # out=torch.einsum('ijkl,mk->ijml', out, self.adj)
            embeddings.append(out)
            # nodes_data = history_data[..., 0]
            # nodes_data_output = self.emdedding_node(nodes_data.long())
            # embeddings.append(nodes_data_output)

        if self.if_time_in_day:
            time_in_day_data = history_data[..., 1:2]
            out=self.time_in_day_emb(time_in_day_data)
            embeddings.append(out)
        # if self.if_day_in_week:
        #     day_in_week_data = history_data[..., 2]
        #     day_in_week_emb = self.day_in_week_emb(day_in_week_data)
        #     embeddings.append(day_in_week_emb)
        # embeddings.append(out_adj)
        # Concatenate all available embeddings
        # if len(embeddings) > 1:
        hidden = torch.cat(embeddings, dim=1)
        # encoding
        # hidden=nn.ReLU()(hidden)
        hidden = self.encoder(hidden)
        # regression
        if self.gcn_bool:
            out_adj=self.gcn(history_data,self.adj)
            hidden=torch.cat([hidden,out_adj],dim=1)
        prediction = self.regression_layer(hidden)
        if self.if_time_in_day:
            time_in_day_data = history_data[..., 1:2]
            out=self.time_in_day_emb(time_in_day_data)
            out=self.time_in_day_emb2(out)
            prediction+=out
        # prediction=torch.einsum('ijkl,mk->ijml', prediction, self.adj)
       
        # if self.is_Smooth:
        #     outputs=exponential_smoothing_4d(outputs,0.9)

        return prediction
