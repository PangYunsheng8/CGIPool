import torch
import os
from torch.utils.data import random_split, Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
from torch_scatter import scatter_add


def degree_as_tag(dataset):
    all_features = []
    all_degrees = []
    tagset = set([])

    for i in range(len(dataset)):
        edge_weight = torch.ones((dataset[i].edge_index.size(1),))
        degree = scatter_add(edge_weight, dataset[i].edge_index[0], dim=0)
        degree = degree.detach().numpy().tolist()
        tagset = tagset.union(set(degree))
        all_degrees.append(degree)
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for i in range(len(dataset)):
        node_features = torch.zeros(len(all_degrees[i]), len(tagset))
        node_features[range(len(all_degrees[i])), [tag2index[tag] for tag in all_degrees[i]]] = 1
        all_features.append(node_features)
    return all_features, len(tagset)


def build_data_list(dataset):
    node_features, num_features = degree_as_tag(dataset)  

    data_list = []
    for i in range(len(dataset)):
        old_data = dataset[i]
        new_data = Data(x=node_features[i], edge_index=old_data.edge_index, y=old_data.y)
        data_list.append(new_data)
    return data_list, num_features


def build_loader(args):
    dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
    args.num_classes = dataset.num_classes
    if args.dataset in ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'ENZYMES']:
        dataset, num_features = build_data_list(dataset)
        args.num_features = num_features
    else:
        args.num_features = dataset.num_features

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, val_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    dataset = TUDataset(os.path.join('data', 'IMDB-BINARY'), name='IMDB-BINARY')
    dataset, nf = build_data_list(dataset)

    for i in range(len(dataset)):
        # print(dataset[i])
        print(dataset[i].x.size())
        # print(dataset[i].y)
        # print(dataset[i].edge_index)
