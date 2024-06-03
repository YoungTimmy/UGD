import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
from torch_geometric.datasets import Planetoid,Coauthor,Amazon
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import argparse
import torch
from deeprobust.graph.global_attack import PRBCD
from torch_geometric.data import Data
import numpy as np
import random
parser = argparse.ArgumentParser()
parser.add_argument('--ptb_rate', type=float, default=0.1, help='perturbation rate.')
args = parser.parse_args()

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
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

seed=1
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
random.seed(seed) 	
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name='pubmed'
#pubmed
if data_name=='pubmed':
    dataset= Planetoid('./data/', 'pubmed')
    data_dict=dataset[0]
    # dataset.transform = T.NormalizeFeatures()
    print(data_dict)
    data = Data(x=data_dict['x'], edge_index=data_dict['edge_index'], y=data_dict['y'])
    data.train_mask = data_dict['train_mask']
    data.val_mask = data_dict['val_mask']
    data.test_mask = data_dict['test_mask']
    print(data.edge_index.shape)

    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    print(data.edge_index.shape)

elif data_name=='CS':
    dataset = Coauthor('./data/', 'CS')
    data=dataset[0]
    data=random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)

elif data_name=='Computers':
    dataset = Amazon('./data/', 'Computers')
    data=dataset[0]
    data=random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)

print(data)

agent = PRBCD(data, device=device)
edge_index, edge_weight = agent.attack(ptb_rate=args.ptb_rate)

print(edge_index.shape)
edge_index=np.array(edge_index.cpu())

print(edge_index)
np.save('./data/'+data_name+'_injection/'+data_name+'_noise_s_'+str(args.ptb_rate)+"_seed_"+str(seed),edge_index)


