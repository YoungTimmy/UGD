import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
from torch_geometric.datasets import Planetoid,Coauthor,Amazon
import torch
import argparse
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool).cuda()
    mask[index] = 1
    return mask

def random_coauthor_amazon_splits(data, num_classes, seed=1):
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


##read the feature matrix and inject the anomal
##after the inkection, the anomal dataset will be saved as npy file for further processing

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='name of dataset (default: cora)')
    parser.add_argument('--compare_num', type=int, default=500,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 0)')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataname = args.dataset
    rootname = './data/'
    if dataname.lower() == 'cora' or dataname.lower() == 'citeseer' or dataname.lower() == 'pubmed':
        dataset = Planetoid(root=rootname, name=dataname)
        data = dataset[0]
    elif dataname=='CS':
        dataset = Coauthor('./data/', 'CS')
        data=dataset[0]
        data=random_coauthor_amazon_splits(data, num_classes=dataset.num_classes)
    elif dataname=='Computers':
        dataset = Amazon('./data/', 'Computers')
        data=dataset[0]
        data=random_coauthor_amazon_splits(data, num_classes=dataset.num_classes)
    print(data)
    ##feature matrix
    if dataname.lower() in ['cora','citeseer','pubmed']:
        num_nodes = data['x'].shape[0]
        feat_mat = np.array(data['x'])
    elif dataname.lower() in ['cs','computers']:
        num_nodes=data.x.shape[0]
        feat_mat=np.array(data.x)
    else:
        print('Wrong data name!')
        assert False
    
    out_feat = np.array(feat_mat, copy=True)
    print(out_feat.shape)
    ratio_list = [0.5]
    for ratio in ratio_list:
        num_anomal = int(num_nodes*ratio)
        anomal_list = np.random.randint(0,num_nodes,num_anomal)
        switched_node = np.zeros((num_anomal,2))
        ##randomly select 50 nodes for distance measuring
        count = 0
        for i in anomal_list:
            tmp_node = np.random.randint(0, num_nodes, args.compare_num)
            max_dist=0
            for j in tmp_node:
                tmp_dist = np.sum((feat_mat[i]-feat_mat[j])**2)
                # print("dist:",tmp_dist)
                if tmp_dist>max_dist:
                    max_dist = tmp_dist
                    max_node = j
        # print("switching node",str(i),"and node", str(j))
            switched_node[count,0] = i
            switched_node[count,1] = j
            count +=1
            out_feat[i] = feat_mat[j]
        save_pth="./data/"+str(dataname)+"_injection/"
        os.makedirs(save_pth,exist_ok=True)
        np.save(save_pth+str(dataname)+"_noise_f_"+str(ratio)+"_seed_"+str(args.seed),out_feat)