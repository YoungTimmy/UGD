import torch
import numpy as np
from torch_geometric.nn import GCN
import torch.nn.functional as F
from util import recon,smooth_loss


class Denoise(torch.nn.Module):
    def __init__(self,dim,alpha=1,beta=0,gamma=0.001,th=0.7,epochs=800,\
                 flag_structure=False,flag_feature=False,\
                    stopnum=0,num_init=100,num=20,dataname='cora'):
        super(Denoise, self).__init__()
        self.encoder = GCN(in_channels=dim,
                            hidden_channels=64,
                            num_layers=2,
                            out_channels=64,
                            dropout=0.,
                            act=torch.nn.functional.relu
                            )
        self.attr_decoder = GCN(in_channels=64,
                            hidden_channels=64,
                            num_layers=2,
                            out_channels=dim,
                            dropout=0.,
                            act=torch.nn.functional.relu
                        )
        self.dataname=dataname
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.th=th
        self.max_th=th+0.02

        self.epochs=epochs
        self.e_step_flag=flag_structure

        self.loss_lst=[]
        self.end_flag=False
        self.only_structure=False
        self.only_feature=False
        
        self.num=num_init
        self.nxt_num=num
        self.stopnum=stopnum
        self.cnt_e=0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if flag_feature==False and flag_structure==True:
            self.only_structure=True
        if flag_feature==True and flag_structure==False:
            self.only_feature=True
        
        
    def m_step_stop(self,loss,epoch):
        self.loss_lst.append(loss)
        if len(self.loss_lst)>self.num and loss>= np.mean(np.array(self.loss_lst[-self.num:])):
            print('Epoch:',epoch)
            print('m_step stop!')
            self.e_step_flag=True
            self.loss_lst=[]
            self.num=self.nxt_num
            if self.only_feature:
                self.end_flag=True
       
        
    def forward(self,x,f,s,epoch=0):
        #E step
        f=f.to(self.device)
        s=s.to(self.device)
        if self.e_step_flag:
            # tmp_s=edge_index
            tmp_s=s.clone()
            if self.cnt_e==0:
                th_now=0.5
            else:
                th_now=self.th
                if self.th<self.max_th:
                    self.th=self.th+0.005
            
            print(th_now)
            adj = torch.sparse_coo_tensor(tmp_s, torch.ones(tmp_s.shape[1]).to(self.device), (x.shape[0], x.shape[0]))
            deg = (torch.sparse.sum(adj, dim=1).to_dense()).unsqueeze(1)
            deg = deg.to(adj.dtype).to(self.device) + 1e-8
            adj_coo = adj.coalesce()
            P = adj_coo @ f
            P /= deg

            edge_nums=int(tmp_s.shape[1]/2)
            
            if self.dataname=='CS':
                del adj,adj_coo,deg
                P_edges=P[tmp_s[0][0:edge_nums]]
                x_edges=f[tmp_s[1][0:edge_nums]]
                w1=F.cosine_similarity(P_edges,x_edges,dim=1) 
                w1=torch.sigmoid(w1)
                del P_edges,x_edges

                P_edges=P[tmp_s[0][edge_nums:]]
                x_edges=f[tmp_s[1][edge_nums:]]
                w2=F.cosine_similarity(P_edges,x_edges,dim=1) 
                w2=torch.sigmoid(w2)
                del P_edges,x_edges,P

                w=torch.cat((w1,w2))
                del w1,w2
            else:
                P_edges=P[tmp_s[0]]
                x_edges=f[tmp_s[1]]
                w=F.cosine_similarity(P_edges,x_edges,dim=1) 
                w=torch.sigmoid(w)  
                     
            remove_idx = torch.where(w < th_now)[0]
            remove_idx = torch.cat([remove_idx, (remove_idx + edge_nums) % tmp_s.shape[1]])
            remove_idx = torch.unique(remove_idx)

            mask = torch.ones(tmp_s.shape[1], dtype=torch.bool)
            mask[remove_idx] = False
            tmp_s=tmp_s[:,mask]

            print(remove_idx.shape)
            print(tmp_s.shape)

            if remove_idx.shape[0]<=self.stopnum and self.cnt_e>3:
                self.end_flag=True
                print('Iteration number:',self.cnt_e)
            
            if self.only_structure and self.cnt_e>=1:
                self.end_flag=True

            s=tmp_s
            self.e_step_flag=False
            self.cnt_e+=1

        #M Step
        emb = self.encoder(x, s)
        beta=self.beta
        x_new = beta*x + (1-beta)*self.attr_decoder(emb,s)
        
        sm_loss=smooth_loss(x_new,s)
        score=torch.mean(recon(x,x_new))

        loss=self.alpha*score+self.gamma*sm_loss
        self.m_step_stop(loss.item(),epoch)
        
        return loss,x_new,s,self.end_flag

    

class Net(torch.nn.Module):
    def __init__(self, conv ,conv_type):
        super(Net, self).__init__()
        self.GConv = conv
        self.reset_parameters()
        self.drop1 = torch.nn.Dropout(0.5)
        self.conv_type = conv_type
        self.drop = False
    def reset_parameters(self):
        for conv in self.GConv:
            conv.reset_parameters()
    def forward(self, data, structure ,attention=False):
        data = data.to(torch.float32).cuda()
        structure = structure.cuda()
        x = self.GConv[0](data ,structure)
        if self.conv_type.lower() == 'gat':
            x = F.elu(x)
        elif self.conv_type.lower() == ('gcn' or 'ufg_s'):
            x = F.relu(x)
        x = self.drop1(x)
        x = self.GConv[1](x, structure)
        return F.log_softmax(x, dim=-1)