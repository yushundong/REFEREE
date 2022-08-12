import argparse
import os
import pickle
import random
import shutil
import time
import copy
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.sparse as sp
from tensorboardX import SummaryWriter
import configs
import gengraph
import util
import models
import warnings
import json
from tqdm import tqdm
warnings.filterwarnings("ignore")

def fair_metric(output,x,labels,idx):
    sens=x[:,-1].squeeze().detach()
    val_y = labels.cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx]==0
    idx_s1 = sens.cpu().numpy()[idx]==1
    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)
    pred_y=torch.argmax(output,-1).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))
    return [parity,equality]

def feature_norm(features):
    min_values = features.min(axis=0,keepdims=True)
    max_values = features.max(axis=0,keepdims=True)
    return 2*(features - min_values)/(max_values-min_values) - 1

def train_node_classifier(G, labels, model, args):

    num_nodes = G.number_of_nodes()
    num_train = int(num_nodes * args.train_ratio)
    num_valid=int(num_nodes*args.valid_ratio)

    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    valid_idx=idx[num_train:num_train+num_valid]
    test_idx = idx[num_train+num_valid:]

    data = gengraph.preprocess_input_graph(G, labels)
    labels_total = torch.tensor(data["labels"], dtype=torch.long)
    labels_train = torch.tensor(data["labels"][:, train_idx], dtype=torch.long)


    data['feat'][0][:,:-1]=feature_norm(data['feat'][0][:,:-1])
    adj = torch.tensor(data["adj"], dtype=torch.float).cuda()
    
    x = torch.tensor(data["feat"], requires_grad=False, dtype=torch.float).cuda()
    optimizer = util.build_optimizer(args, model.parameters(), weight_decay=args.weight_decay)

    total_begin_time=time.time()
    valid_acc=0

    ypred = None
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        model.zero_grad()

        ypred, _ = model(x, adj)
        ypred_train = ypred[:, train_idx, :]

        loss = model.loss(ypred_train, labels_train.cuda())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        result_train, result_valid = evaluate_node(ypred.cpu(), data["labels"], train_idx, valid_idx)

        if result_valid["acc"]>valid_acc:
            train_acc = result_train['acc']
            valid_acc=result_valid["acc"]
            model.eval()
            ypred, _ = model(x, adj)
            cg_data = {"adj": data["adj"], "feat": data["feat"], "label": data["labels"], "pred": ypred.cpu().detach().numpy(), "train_idx": train_idx+valid_idx,}
            model_state_dict = copy.deepcopy(model.state_dict())
            result_train, result_test = evaluate_node(ypred.cpu(), data["labels"], train_idx, test_idx)
            test_acc=result_test['acc']

            optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

    os.makedirs('./ckpt', exist_ok=True)
    filename='./ckpt/'+args.dataset+'_'+args.method+'.pth.tar'

    if not args.remove:
        torch.save({"epoch": -1, "model_type": args.method, "optimizer": optimizer, "model_state": model_state_dict, "optimizer_state": optimizer_state_dict, "cg": cg_data}, filename, pickle_protocol=4)

    return [fair_metric(ypred.squeeze(), x[0], labels_total[0], list(range(num_nodes))) + [
        train_acc.item()*0.8+valid_acc.item()*0.1+test_acc.item()*0.1] + [time.time() - total_begin_time]]


def evaluate_node(ypred, labels, train_idx, test_idx):
    _, pred_labels = torch.max(ypred, 2)
    pred_labels = pred_labels.numpy()

    pred_train = np.ravel(pred_labels[:, train_idx])
    pred_test = np.ravel(pred_labels[:, test_idx])
    labels_train = np.ravel(labels[:, train_idx])
    labels_test = np.ravel(labels[:, test_idx])

    result_train = {"prec": metrics.precision_score(labels_train, pred_train, average="macro"), "recall": metrics.recall_score(labels_train, pred_train, average="macro"), "acc": metrics.accuracy_score(labels_train, pred_train), "conf_mat": metrics.confusion_matrix(labels_train, pred_train),}
    result_test = {"prec": metrics.precision_score(labels_test, pred_test, average="macro"), "recall": metrics.recall_score(labels_test, pred_test, average="macro"), "acc": metrics.accuracy_score(labels_test, pred_test), "conf_mat": metrics.confusion_matrix(labels_test, pred_test),}
    return result_train, result_test


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh*max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map =  np.array(idx_map)
    return idx_map
def load_credit(dataset, sens_attr="Age", predict_attr="NoDefaultNextMonth", path="./dataset/credit/"):
    from scipy.spatial import distance_matrix

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    ##################################
    header.remove(sens_attr)
    header.append(sens_attr)
    ##################################

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    
    return adj, features, labels, edges, sens  

def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="../dataset/bail/"):

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    
    ##################################
    header.remove(sens_attr)
    header.append(sens_attr)
    ##################################

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    
    return adj, features, labels, edges, sens

def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="../dataset/german/"):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    ##################################
    header.remove('Gender')
    header.append('Gender')
    ##################################

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    return adj, features, labels, edges, sens
    
def task_real_data(args):

    if args.dataset=='german':
        path = "./dataset/german"
        adj, features, labels, edges, sens = load_german('german', path=path)
    elif args.dataset=='credit':
        path = "./dataset/credit"
        adj, features, labels, edges, sens = load_credit('credit', path=path)
    elif args.dataset=='bail':
        path = "./dataset/bail"
        adj, features, labels, edges, sens = load_bail('bail', path=path)

    G=nx.from_numpy_matrix(adj.todense())
    feat_dict = {i:{'feat': np.array(features[i], dtype=np.float32)} for i in G.nodes()}

    if args.remove:

        configs_loaded = json.load(open('./configs/configs.json'))[
            '{}_{}_{}'.format(args.dataset,args.explainer_backbone,args.debias_ratio)]
        args.lr, args.num_epochs,args.select=configs_loaded['lr'],configs_loaded['num_epochs'],configs_loaded['select']

        if args.debias_ratio>0:
            node_list=np.random.choice(list(range(1000)),size=args.debias_ratio*10,replace=False).tolist()

            total_unfair_edge=set()
            for j,i in enumerate(node_list):
                neighbors=np.load('./log/{}_{}_GAT/neighbors_node_idx_{}.npy'.format(args.dataset,args.explainer_backbone,i))
                neighbors=neighbors[:-1]

                masks=np.load('./log/{}_{}_GAT/masked_adj_node_idx_{}.npy'.format(args.dataset,args.explainer_backbone,i))
                wass_mask=masks[1]
                unwass_mask=masks[2]
                wass_mask=np.triu(wass_mask)
                unwass_mask=np.triu(unwass_mask)

                index=np.argsort(wass_mask.flatten())[-args.select:]
                wass_mask_new=np.zeros(wass_mask.shape).flatten()
                wass_mask_new[index]=1

                index=np.argsort(unwass_mask.flatten())[-args.select:]
                unwass_mask_new=np.zeros(unwass_mask.shape).flatten()
                unwass_mask_new[index]=1

                wass_mask=np.argwhere(wass_mask_new.reshape(wass_mask.shape)>0).tolist()
                unwass_mask=np.argwhere(unwass_mask_new.reshape(unwass_mask.shape)>0).tolist()

                temp=set()
                for one in unwass_mask:
                    a=(neighbors[one[0]],neighbors[one[1]])
                    temp.add(a)

                temp_wass=set()
                for one in wass_mask:
                    a=(neighbors[one[0]],neighbors[one[1]])
                    temp_wass.add(a)

                unwass_mask=set(temp).difference(set(temp_wass))

                total_unfair_edge.update(unwass_mask)

            for edge in total_unfair_edge:
                try:
                    G.remove_edge(edge[0],edge[1])
                except:
                    continue

    nx.set_node_attributes(G, feat_dict)
    args.input_dim=features.shape[-1]
    model = models.GNN(args.input_dim, args.hidden_dim, args.output_dim, 2, args.num_gc_layers, bn=args.bn, args=args)
    
    if args.gpu:
        model = model.cuda()

    return train_node_classifier(G, labels, model, args)

def main():

    seed=100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    prog_args = configs.arg_parse()

    print('Start fitting ...')
    result=task_real_data(prog_args)
    print('Fitting completed.')
    if prog_args.remove:
        result_print=pd.DataFrame([{'dataset':prog_args.dataset, 'explainer_backbone':prog_args.explainer_backbone, 'GNN_type':prog_args.method,
                          'SP':result[0][0], 'EO': result[0][1], 'Acc':result[0][2], 'Training Time':result[0][3]
                          }])
    else:
        result_print = pd.DataFrame([{'dataset': prog_args.dataset,
                                      'GNN_type': prog_args.method,
                                        'Acc': result[0][2],
                                      'Training Time': result[0][3]}])
    print(result_print)

if __name__ == "__main__":
    main()

