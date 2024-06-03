import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from deeprobust.graph.defense import *
from deeprobust.graph.global_attack import MetaApprox, Metattack
from deeprobust.graph.utils import *
from deeprobust.graph.defense import *
import argparse
from scipy.sparse import csr_matrix
import scipy
from torch_geometric.utils import from_scipy_sparse_matrix,to_scipy_sparse_matrix
from torch_geometric.datasets import Planetoid


def test_f(features,data):
    ''' testing model '''
    classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)

    classifier = classifier.to(device)
    classifier.fit(features, adj, labels, data.idx_train, train_iters=201,
                   idx_val=data.idx_val,
                   idx_test=data.idx_test,
                   verbose=True, attention=attention) # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    classifier.eval()
    idx_test=torch.tensor(data.idx_test)
    acc_test = classifier.test(idx_test)

    return acc_test

def test_s(adj,data):
    ''' testing model '''
    classifier = globals()[args.modelname](nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)

    classifier = classifier.to(device)
    classifier.fit(features, adj, labels, data.idx_train, train_iters=201,
                   idx_val=data.idx_val,
                   idx_test=data.idx_test,
                   verbose=True, attention=attention) # idx_val=idx_val, idx_test=idx_test , model_name=model_name
    classifier.eval()
    idx_test=torch.tensor(data.idx_test)
    acc_test = classifier.test(idx_test)

    return acc_test

def main():
    ''
    """save the mettacked adj"""
 
    model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    
    if structure_attack:

        modified_s = model.modified_adj
        if scipy.sparse.issparse(modified_s)==False:
            modified_s= scipy.sparse.csr_matrix(modified_s.cpu())

        print(modified_s.shape)
        print(adj)
        edge_index_new,_=from_scipy_sparse_matrix(modified_s)
        print(edge_index_new)

        np.save('./data/'+str(args.dataset)+'_injection/'+str(args.dataset)+"_noise_s_"+str(args.ptb_rate)+"_seed_"+str(args.seed),edge_index_new)



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.1,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')

parser.add_argument('--modelname', type=str, default='GCN',  choices=['GCN', 'GAT','GIN', 'JK'])
parser.add_argument('--defensemodel', type=str, default='GCNJaccard',  choices=['GCNJaccard', 'RGCN', 'GCNSVD'])
parser.add_argument('--GNNGuard', type=bool, default=False,  choices=[True, False])

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_attack=False
structure_attack=True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

dataname = args.dataset
rootname = './data/'
if dataname.lower() == 'cora' or dataname.lower() == 'citeseer' or dataname.lower() == 'pubmed':
    dataset = Planetoid(root=rootname, name=dataname)
    data = dataset[0]
print(data)

features=data['x']
labels=data['y']
adj=to_scipy_sparse_matrix(data['edge_index'])
idx_train=torch.where(data['train_mask']==True)[0]
idx_val=torch.where(data['val_mask']==True)[0]
idx_test=torch.where(data['test_mask']==True)[0]


idx_unlabeled = np.union1d(idx_val, idx_test)
print("type:",type(adj),type(features),type(labels))
if scipy.sparse.issparse(features)==False:
    features = scipy.sparse.csr_matrix(features)


perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
#1. to CSR sparse
adj, features = csr_matrix(adj), csr_matrix(features)

adj = adj + adj.T
adj[adj>1] = 1

# Setup GCN as the Surrogate Model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
        dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=201)

# Setup Attack Model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, attack_structure=structure_attack, attack_features=feature_attack, device=device, lambda_=lambda_)

else:
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape,  attack_structure=structure_attack, attack_features=feature_attack, device=device, lambda_=lambda_)

model = model.to(device)
attention = args.GNNGuard # if True, our method; if False, run baselines
main()

