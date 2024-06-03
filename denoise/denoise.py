import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
from torch_geometric.nn import GCNConv
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import os
import torch.nn.functional as F
import argparse
from util import load_data,set_seed
from models import Denoise,Net
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
def main(args):
    lr=args.lr                #learning rate of denoising stage
    epochs=args.epochs        #epochs of denoising stage
    cls_epochs=100            #epochs of classification stage
    set_seed(args.seed)

    flag_feature=args.flag_f    # denoise or not
    flag_structure=args.flag_s  # denoise or not

    flag=flag_feature or flag_structure

    alpha=1.0             #feature reconstruction loss
    gamma=args.gamma      #smooth loss
    data_name=args.dataset

    if data_name in ['cora','citeseer','pubmed']:
        data_dict=load_data(data_name,args.seed)
        data = Data(x=data_dict['x'], edge_index=data_dict['edge_index'], y=data_dict['y'])
        data.train_mask = data_dict['train_mask']
        data.val_mask = data_dict['val_mask']
        data.test_mask = data_dict['test_mask']

    elif data_name in ['Computers','CS']:
        data=load_data(data_name,args.seed)

    threshold=args.th

    f_ratio=args.f_ratio
    if f_ratio>0:
        noise_data_path='./data/'+str(data_name)+'_injection/'+str(data_name)+'_noise_f_'+str(f_ratio)+'_seed_'+str(args.seed)+'.npy'
        noise_feature=np.load(noise_data_path)
        data.x=torch.tensor(noise_feature)

    s_ratio=args.s_ratio
    if s_ratio>0:
        noise_data_path='./data/'+str(data_name)+'_injection/'+str(data_name)+'_noise_s_'+str(s_ratio)+'_seed_'+str(args.seed)+'.npy'
        noise_structure=np.load(noise_data_path)
        data.edge_index=torch.tensor(noise_structure)

    print(data.edge_index.shape)

    all_edges=zip(data.edge_index[0].tolist(),data.edge_index[1].tolist())
    G = nx.from_edgelist(all_edges)
    lst1,lst2=zip(*G.edges)
    new_edges=torch.tensor([list(lst1),list(lst2)])
    new_edges = torch.cat([new_edges, new_edges.flip(0)], dim=1)
    data.edge_index=new_edges

    del all_edges,G,lst1,lst2,new_edges

    if flag==True:
        data=data.to(device)
        model=Denoise(dim=data.x.shape[1],alpha=alpha,beta=args.beta,gamma=gamma,\
                      th=threshold,epochs=args.epochs,\
                      flag_structure=flag_structure,flag_feature=flag_feature,\
                      stopnum=args.stopnum,num_init=args.ave_num_init,num=args.ave_num,\
                      dataname=data_name).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.)
        f=data.x.clone()
        s=data.edge_index.clone()

        print("Start Train!")
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss,x_new,edge_index_new,end_flag= model.forward(data.x,f,s,epoch=epoch+1)
            f=x_new.detach().cpu()
            s=edge_index_new.detach().cpu()
            loss.backward()
            optimizer.step()
            loss= loss.item()

            if (epoch+1)%10==0:
                print("Epoch: ",epoch+1," Loss: ",round(loss,5))
                print("----------------------------------")

            if end_flag==True:
                print("Finished! ")
                break

        print(s.shape)
        data.edge_index_new=s
        data.x_new=f

    #classification
        
    data=data.to(device)
    num_classes=data.y.max().item()+1

    cls_model=Net(torch.nn.ModuleList([GCNConv(data.x.shape[1], 256, add_self_loops=True),
                                   GCNConv(256, num_classes, add_self_loops=True)]).to(device),'gcn')
    
    cls_optimizer = torch.optim.Adam(cls_model.parameters(), lr=0.01, weight_decay=1e-3)

    def train():
        cls_model.train()
        cls_optimizer.zero_grad()
        if flag_feature==True:
            x=data.x_new
        else:
            x=data.x

        if flag_structure==True:
            edge=data.edge_index_new
        else:
            edge=data.edge_index

        out = cls_model(x, edge)
        cls_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        cls_loss.backward()
        cls_optimizer.step()
        return float(cls_loss)
    @torch.no_grad()
    def test():
        cls_model.eval()
        if flag_feature==True:
            x=data.x_new
        else:
            x=data.x
            
        if flag_structure==True:
            edge=data.edge_index_new
        else:
            edge=data.edge_index
            
        out = cls_model(x, edge)

        pred = out.argmax(dim=-1)
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

        return accs

    best_val_acc = final_test_acc = 0
    for epoch in range(1, cls_epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            if test_acc>final_test_acc:
                final_test_acc=test_acc
            
    print('Final test accuracy:{}'.format(final_test_acc))

    return final_test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    parser.add_argument("-d", "--dataset", type=str, default='citeseer')
    parser.add_argument("--th", default=0.7, type=float)
    parser.add_argument("--beta", default=0, type=float)
    parser.add_argument("--gamma", default=1e-3, type=float)
    parser.add_argument("--epochs", default=5000, type=int)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--flag_f", default=True,type=bool)
    parser.add_argument("--flag_s", default=True,type=bool)
    parser.add_argument("--f_ratio", default=0.5,type=float)
    parser.add_argument("--s_ratio", default=0.1,type=float)
    parser.add_argument("--lr", default=1e-3,type=float)
    parser.add_argument("--ave_num_init", default=60,type=float)
    parser.add_argument("--ave_num", default=20,type=float)
    parser.add_argument("--stopnum", default=0,type=float)
    args = parser.parse_args()
    print(args)
    main(args)





