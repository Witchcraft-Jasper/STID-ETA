import torch
from torch import nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class graphConvolution(nn.Module):
    def __init__(self, in_features, out_features,hidden=32, bias=True):
        super(graphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.conv1 = nn.Conv2d(in_features, hidden, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(hidden, out_features, kernel_size=(1, 1), stride=(1, 1))
        self.dropout=0.1
        self.batch_norm=nn.BatchNorm2d(out_features)
        # self.adj_embed=nn.Embedding(5, 20)
        # self.fc=nn.Linear(1, 1)
    def forward(self, input, adj):
        # input=input.permute(0,3,1,2)
        # adj=self.adj_embed(adj)
        out=torch.einsum('ijkl,mk->ijml', input, adj)
        out=self.conv1(out)
        out=self.conv2(out)
        # out=out.permute(0,2,3,1)
        # output = F.relu(torch.einsum('ijkl,mk->ijml', out, adj))
        output = self.batch_norm(out)
        # output = self.fc(output)
        return output

# class graphConvolution(nn.Module):
#     def __init__(self, in_features, out_features,hidden=32, bias=True):
#         super(graphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.empty(in_features, out_features))
#         nn.init.xavier_uniform_(self.weight)
#         self.conv1 = nn.Conv2d(in_features, hidden, kernel_size=(1, 1), stride=(1, 1))
#         self.conv2 = nn.Conv2d(hidden, out_features, kernel_size=(1, 1), stride=(1, 1))
#         self.fc=nn.Linear(1, 1)
#     def forward(self, input, adj):
#         # input=input.permute(0,3,1,2)
#         out=self.conv1(input)
#         out=self.conv2(out)
#         # out=out.permute(0,2,3,1)
#         output = F.relu(torch.einsum('ijkl,mk->ijml', out, adj))
#         output = self.fc(output)
#         return output
    
    
class polyFit(nn.Module):
    def __init__(self,num=3):
        super(polyFit, self).__init__()
        self.params = torch.randn(3, requires_grad=True)
    def forward(self, x):
        y_pred =  self.params[0]*x**2 +  self.params[1]*x +  self.params[2]
        y_pred=nn.ReLU()(y_pred)
        return y_pred ** 0.5
    

