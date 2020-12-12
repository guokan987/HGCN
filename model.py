import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


import numpy as np
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d

from utils import ST_BLOCK_0 #ASTGCN
from utils import ST_BLOCK_1 #DGCN_Mask/DGCN_Res
from utils import ST_BLOCK_2_r #DGCN_recent

from utils import ST_BLOCK_4 #Gated-STGCN
from utils import ST_BLOCK_5 #GRCN
from utils import ST_BLOCK_6 #OTSGGCN
from utils import multi_gcn #gwnet
from utils import GCNPool #H_GCN
from utils import Transmit
from utils import gate
from utils import GCNPool_dynamic
from utils import GCNPool_h
from utils import T_cheby_conv_ds_1
from utils import dynamic_adj
from utils import SATT_h_gcn
from sparse_activations import Sparsemax
"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class ASTGCN_Recent(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(ASTGCN_Recent,self).__init__()
        self.block1=ST_BLOCK_0(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,12,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x)#b,12,n,1
        return x,d_adj,t_adj
    

    
class DGCN_recent(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3): 
        super(DGCN_recent,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_2_r(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_2_r(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        
        self.conv1=Conv2d(dilation_channels,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        
        self.supports=supports
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
    def forward(self,input):
        x=input
    
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=F.dropout(A,0.5,self.training)
              
        x,_,_=self.block1(x,A1)
        x,d_adj,t_adj=self.block2(x,A1)
    
        x=self.conv1(x).permute(0,3,2,1).contiguous()#b,c,n,l
        return x,d_adj,A


class LSTM(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(LSTM,self).__init__()
        self.lstm=nn.LSTM(in_dim,dilation_channels,batch_first=True)#b*n,l,c
        self.c_out=dilation_channels
        tem_size=length
        self.tem_size=tem_size
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        
        
    def forward(self,input):
        x=input
        shape = x.shape
        h = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        c = Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        hidden=(h,c)
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
        x,hidden=self.lstm(x,hidden)
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l
        return x,hidden[0],hidden[0]

class GRU(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(GRU,self).__init__()
        self.gru=nn.GRU(in_dim,dilation_channels,batch_first=True)#b*n,l,c
        self.c_out=dilation_channels
        tem_size=length
        self.tem_size=tem_size
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1,tem_size),
                          stride=(1,1), bias=True)
        
    def forward(self,input):
        x=input
        shape = x.shape
        h =Variable(torch.zeros((1,shape[0]*shape[2],self.c_out))).cuda()
        hidden=h
        
        x=x.permute(0,2,3,1).contiguous().view(shape[0]*shape[2],shape[3],shape[1])
        x,hidden=self.gru(x,hidden)
        x=x.view(shape[0],shape[2],shape[3],self.c_out).permute(0,3,1,2).contiguous()
        x=self.conv1(x)#b,c,n,l
        return x,hidden[0],hidden[0]
        
class Gated_STGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(Gated_STGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        x=self.conv1(x)#b,12,n,1
        return x,adj,adj 


class GRCN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(GRCN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_5(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        
        adj=self.supports[0]
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.conv1(x)
        return x,adj,adj     

class OGCRNN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5)
        x=self.block1(x,A)
        
        x=self.conv1(x)
        return x,A,A   

#OTSGGCN    
class OTSGGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OTSGGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        self.block1=ST_BLOCK_6(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_6(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=torch.eye(self.num_nodes).cuda()-A
        A1=F.dropout(A1,0.5)
        x=self.block1(x,A1)
        x=self.block2(x,A1)
        
        x=self.conv1(x)#b,12,n,1
        return x,A1,A1     
    
    
#gwnet    
class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1,out_dim=12,residual_channels=32,
                 dilation_channels=32,skip_channels=256,
                 end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
            



        
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                
                new_dilation *=2
                receptive_field += additional_scope
                
                additional_scope *= 2
                
                self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1=BatchNorm2d(in_dim,affine=False)

    def forward(self, input):
        input=self.bn_1(input)
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)           

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,adp,adp

class H_GCN_wh(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN_wh, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        
        self.supports = supports
        
        self.supports_len = 0
        
        if supports is not None:
            self.supports_len += len(supports)
        
        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)   
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
        self.supports_len +=1
        
        
        
        Kt1=2
        self.block1=GCNPool(dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        

    def forward(self, input):
        x=self.bn(input)
        shape=x.shape
        
        if self.supports is not None:
            #nodes
            #A=A+self.supports[0]
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
            
            
        skip=0
        x = self.start_conv(x)
        
        #1
        x=self.block1(x,new_supports)
        
        s1=self.skip_conv1(x)
        skip=s1+skip
        
        #2
        x=self.block2(x,new_supports)
       
        s2=self.skip_conv2(x)
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip
                
        #output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x,x,A

class H_GCN_wdf(nn.Module):
    def __init__(self,device, num_nodes, cluster_nodes,dropout=0.3, supports=None,supports_cluster=None,transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN_wdf, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.cluster_nodes=cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_cluster = supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        
        
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)
        
        
       

    def forward(self, input, input_cluster):
        x=self.bn(input)
        shape=x.shape
        input_c=input_cluster
        x_cluster=self.bn_cluster(input_c)
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster = self.supports_cluster + [A_cluster]
        
        #network
        transmit=self.transmit              
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        
        x_1=(torch.einsum('mn,bcnl->bcml',transmit,x_cluster))   
        
        x=self.gate1(x,x_1)
        
       
        skip=0
        skip_c=0
        #1
        x_cluster=self.block_cluster1(x_cluster,new_supports_cluster) 
        x=self.block1(x,new_supports)   
        
        x_2=(torch.einsum('mn,bcnl->bcml',transmit,x_cluster)) 
        
        x=self.gate2(x,x_2) 
        
        
        s1=self.skip_conv1(x)
        skip=s1+skip 
        
       
        #2       
        x_cluster=self.block_cluster2(x_cluster,new_supports_cluster)
        x=self.block2(x,new_supports) 
        
        x_3=(torch.einsum('mn,bcnl->bcml',transmit,x_cluster)) 
        
        x=self.gate3(x,x_3)
           
        
        s2=self.skip_conv2(x)      
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip        
        
       
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        return x,A,A    
    
    
class H_GCN(nn.Module):
    def __init__(self,device, num_nodes, cluster_nodes,dropout=0.3, supports=None,supports_cluster=None,transmit=None,length=12, 
                 in_dim=1,in_dim_cluster=3,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(H_GCN, self).__init__()
        self.dropout = dropout
        self.num_nodes=num_nodes
        self.transmit=transmit
        self.cluster_nodes=cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.supports_cluster = supports_cluster
        
        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster+=len(supports_cluster)

        
        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster=Parameter(torch.zeros(cluster_nodes,cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len +=1
        self.supports_len_cluster +=1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)  
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10,cluster_nodes).to(device), requires_grad=True).to(device)  
        
        
        self.block1=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-6,3,dropout,num_nodes,
                            self.supports_len)
        self.block2=GCNPool(2*dilation_channels,dilation_channels,num_nodes,length-9,2,dropout,num_nodes,
                            self.supports_len)
        
        self.block_cluster1=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-6,3,dropout,cluster_nodes,
                            self.supports_len)
        self.block_cluster2=GCNPool(dilation_channels,dilation_channels,cluster_nodes,length-9,2,dropout,cluster_nodes,
                            self.supports_len)
        
        self.skip_conv1=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        self.skip_conv2=Conv2d(2*dilation_channels,skip_channels,kernel_size=(1,1),
                          stride=(1,1), bias=True)
        
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        
        
        
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.conv_cluster1=Conv2d(dilation_channels,out_dim,kernel_size=(1,3),
                          stride=(1,1), bias=True)
        self.bn_cluster=BatchNorm2d(in_dim_cluster,affine=False)
        self.gate1=gate(2*dilation_channels)
        self.gate2=gate(2*dilation_channels)
        self.gate3=gate(2*dilation_channels)
        
        self.transmit1=Transmit(dilation_channels,length,transmit,num_nodes,cluster_nodes)
        self.transmit2=Transmit(dilation_channels,length-6,transmit,num_nodes,cluster_nodes)
        self.transmit3=Transmit(dilation_channels,length-9,transmit,num_nodes,cluster_nodes)
       

    def forward(self, input, input_cluster):
        x=self.bn(input)
        shape=x.shape
        input_c=input_cluster
        x_cluster=self.bn_cluster(input_c)
        if self.supports is not None:
            #nodes
            A=F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d=1/(torch.sum(A,-1))
            D=torch.diag_embed(d)
            A=torch.matmul(D,A)
            
            new_supports = self.supports + [A]
            #region
            A_cluster=F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c=1/(torch.sum(A_cluster,-1))
            D_c=torch.diag_embed(d_c)
            A_cluster=torch.matmul(D_c,A_cluster)
            
            new_supports_cluster = self.supports_cluster + [A_cluster]
        
        #network
        transmit=self.transmit              
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        transmit1 = self.transmit1(x,x_cluster)
        x_1=(torch.einsum('bmn,bcnl->bcml',transmit1,x_cluster))   
        
        x=self.gate1(x,x_1)
        
       
        skip=0
        skip_c=0
        #1
        x_cluster=self.block_cluster1(x_cluster,new_supports_cluster) 
        x=self.block1(x,new_supports)   
        transmit2 = self.transmit2(x,x_cluster)
        x_2=(torch.einsum('bmn,bcnl->bcml',transmit2,x_cluster)) 
        
        x=self.gate2(x,x_2) 
        
        
        s1=self.skip_conv1(x)
        skip=s1+skip 
        
       
        #2       
        x_cluster=self.block_cluster2(x_cluster,new_supports_cluster)
        x=self.block2(x,new_supports) 
        transmit3 = self.transmit3(x,x_cluster)
        x_3=(torch.einsum('bmn,bcnl->bcml',transmit3,x_cluster)) 
        
        x=self.gate3(x,x_3)
           
        
        s2=self.skip_conv2(x)      
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip        
        
       
        
        #output
        x = F.relu(skip)      
        x = F.relu(self.end_conv_1(x))            
        x = self.end_conv_2(x)              
        return x,transmit3,A
    

    



