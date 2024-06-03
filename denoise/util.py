from torch_geometric.datasets import Planetoid,Coauthor,Amazon
import torch
import numpy as np
import random
from torch_geometric.utils import get_laplacian,to_scipy_sparse_matrix


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask[index] = 1
    return mask

def random_coauthor_amazon_splits(data, num_classes, seed):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0),generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0),generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def load_data(data_name='cora',seed=1):
    if data_name=='citeseer':
        dataset = Planetoid('./data/', 'citeseer')
        data=dataset[0]
    elif data_name=='cora':
        dataset = Planetoid('./data/', 'cora')
        data=dataset[0]
    elif data_name=='pubmed':
        dataset= Planetoid('./data/', 'pubmed')
        data=dataset[0]
    elif data_name=='Computers':
        dataset = Amazon('./data/', 'Computers')
        data=dataset[0]
        data=random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(data)
    elif data_name=='CS':
        dataset = Coauthor('./data/', 'CS')
        data=dataset[0]
        data=random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(data)
    
    else:
        print("Wrong data name!")
        assert False

    return data
        
def set_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    random.seed(seed) 	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def recon(x,x_):  
    l2=torch.sqrt(torch.sum(torch.pow(x - x_, 2),1))
    return l2


def smooth_loss(x_,edge_index,edge_weight=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L,weight=get_laplacian(edge_index=edge_index,num_nodes=x_.shape[0], normalization='sym')
    L = to_scipy_sparse_matrix(L,weight)
    
    L_torch = torch.sparse_coo_tensor(
        torch.LongTensor(np.array([L.row, L.col])),
        torch.FloatTensor(L.data),
        torch.Size(L.shape)
    ).to(device) 
    loss=x_.T @torch.matmul(L_torch,x_)
    loss=torch.trace(loss)
    
    return loss