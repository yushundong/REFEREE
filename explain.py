import math
import time
import os
from scipy.stats import wasserstein_distance
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import util
import tensorboardX.utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x, y, p=0.5, lam=10, its=10, cuda=False):
    if len(x.shape)<2:
        x=x.unsqueeze(0)
    if len(y.shape)<2:
        y=y.unsqueeze(0)
    nx = x.shape[0]
    ny = y.shape[0]
    M = pdist(x, y)
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 0.0 / (nx * ny))
    delta = torch.max(M_drop).cpu().detach()
    eff_lam = (lam / M_mean).cpu().detach()

    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    ainvK = K / a
    u = a
    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()
    upper_t = u * (torch.t(v) * K).detach()
    E = upper_t * Mt
    D = 2 * torch.sum(E)
    if cuda:
        D = D.cuda()
    return D, Mlam

class REFEREE:
    def __init__(self, model, adj, feat, label, pred, train_idx, args):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        if args.dataset!='german':
            self.test_idx=[one for one in list(range(self.feat[0].shape[0])) if one not in self.train_idx]
        else:
            self.test_idx=list(range(self.feat[0].shape[0]))
        self.node_idx=[]

        self.n_hops = args.num_gc_layers
        self.neighborhoods = util.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.print_training = args.debug_mode

        self.explain_true=[]
        self.explain_true_wass=[]
        self.explain_true_minuswass=[]
        self.wass_dis=[]
        self.wass_dis_ori=[]
        self.wass_dis_att=[]
        self.start_wass_dis=[]
        self.start_wass_dis_ori=[]
        self.start_wass_dis_att=[]
        self.start_wass_dis_unfair=[]
        self.wass_dis_unfair=[]

        self.folder_name=self.args.dataset+'_'+self.args.explainer_backbone+'_'+self.args.method
        os.makedirs('log/'+self.folder_name,exist_ok=True)

        self.tensor_pred=torch.tensor(self.pred).cuda().softmax(-1)

    def explain(self, node_idx):
        seed = 100
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(node_idx)
        sub_label = np.expand_dims(sub_label, axis=0)

        sub_adj = np.expand_dims(sub_adj, axis=0)
        sub_feat = np.expand_dims(sub_feat, axis=0)

        adj   = torch.tensor(sub_adj, dtype=torch.float)
        x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).cuda()
        label = torch.tensor(sub_label, dtype=torch.long)

        pred_label = np.argmax(self.pred[0][neighbors], axis=1)


        explainer_fair = ExplainModule(adj=adj, x=x, model=self.model, label=label, args=self.args)
        if self.args.gpu:
            explainer_fair = explainer_fair.cuda()

        explainer_unfair = ExplainModule(adj=adj, x=x, model=self.model, label=label, args=self.args)
        if self.args.gpu:
            explainer_unfair = explainer_unfair.cuda()

        self.model.eval()

        index=self.feat[0][self.test_idx][:,-1]==1-self.feat[0][node_idx][-1]
        new_tensor_pred=self.tensor_pred[0].clone()
        new_index=self.feat[0][self.test_idx][:,-1]==self.feat[0][node_idx][-1]
        wass_dis=wasserstein_distance(self.tensor_pred[0][self.test_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[self.test_idx][new_index][:,0].cpu().detach().numpy())*100000
        self.wass_dis_ori.append(wass_dis)
        wass_dis = wasserstein(
            self.tensor_pred[0][self.test_idx][index],
            new_tensor_pred[self.test_idx][new_index], cuda=True)[0]
        self.start_wass_dis_ori.append(wass_dis.cpu().detach().item())

        for epoch in range(self.args.num_epochs):

            explainer_unfair.optimizer.zero_grad()
            _, _ = explainer_unfair(node_idx_new)
            ypred_unfair, adj_atts = explainer_unfair.cal_WD_ypred(node_idx_new, threshold=self.args.threshold)
            loss = explainer_unfair.loss(ypred_unfair, pred_label, node_idx_new) * self.args.fidelity_unfair_weight

            index=self.feat[0][self.test_idx][:,-1]==1-self.feat[0][node_idx][-1]
            new_tensor_pred=self.tensor_pred[0].clone()
            new_tensor_pred[node_idx]=ypred_unfair
            new_index=self.feat[0][self.test_idx][:,-1]==self.feat[0][node_idx][-1]
            wass_dis=wasserstein(self.tensor_pred[0][self.test_idx][index],new_tensor_pred[self.test_idx][new_index],cuda=True)[0]

            WD_loss_unfair_raw=wass_dis
            WD_loss_unfair=WD_loss_unfair_raw

            fidelity_loss_unfair=loss.item()
            loss -= wass_dis*self.args.WD_unfair_weight

            index=self.feat[0][self.test_idx][:,-1]==1-self.feat[0][node_idx][-1]
            new_tensor_pred=self.tensor_pred[0].clone()
            new_tensor_pred[node_idx]=ypred_unfair
            new_index=self.feat[0][self.test_idx][:,-1]==self.feat[0][node_idx][-1]
            wass_dis=wasserstein_distance(self.tensor_pred[0][self.test_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[self.test_idx][new_index][:,0].cpu().detach().numpy())
            wass_dis_unfair = WD_loss_unfair_raw
            if epoch==0:
                self.start_wass_dis_unfair.append(wass_dis)
            elif epoch+1==self.args.num_epochs:
                self.wass_dis_unfair.append(wass_dis_unfair.cpu().detach().item())

            loss.backward(retain_graph=True)
            explainer_unfair.optimizer.step()

            ###############################################
            explainer_fair.optimizer.zero_grad()

            _, _ = explainer_unfair(node_idx_new)
            _, _ = explainer_fair(node_idx_new)

            ypred, adj_atts = explainer_fair.cal_WD_ypred(node_idx_new, threshold=self.args.threshold)

            loss = explainer_fair.loss(ypred, pred_label, node_idx_new)* self.args.fidelity_fair_weight

            index=self.feat[0][self.test_idx][:,-1]==1-self.feat[0][node_idx][-1]
            new_tensor_pred=self.tensor_pred[0].clone()
            new_tensor_pred[node_idx]=ypred
            new_index=self.feat[0][self.test_idx][:,-1]==self.feat[0][node_idx][-1]

            wass_dis=wasserstein(self.tensor_pred[0][self.test_idx][index],new_tensor_pred[self.test_idx][new_index],cuda=True)[0]

            WD_loss_fair_raw=wass_dis
            WD_loss_fair=WD_loss_fair_raw
            fidelity_loss=loss.item()

            loss += wass_dis*self.args.WD_fair_weight

            index=self.feat[0][self.test_idx][:,-1]==1-self.feat[0][node_idx][-1]
            new_tensor_pred=self.tensor_pred[0].clone()
            new_tensor_pred[node_idx]=ypred
            new_index=self.feat[0][self.test_idx][:,-1]==self.feat[0][node_idx][-1]
            wass_dis=wasserstein_distance(self.tensor_pred[0][self.test_idx][index][:,0].cpu().detach().numpy(),new_tensor_pred[self.test_idx][new_index][:,0].cpu().detach().numpy())
            wass_dis_fair=WD_loss_fair_raw
            if epoch==0:
                self.start_wass_dis.append(wass_dis)
            elif epoch+1==self.args.num_epochs:
                self.wass_dis.append(wass_dis_fair.cpu().detach().item())

            KL_loss=(0.5*F.kl_div(torch.log(explainer_fair.masked_adj.flatten().softmax(-1)),explainer_unfair.masked_adj.flatten().softmax(-1))*+0.5*F.kl_div(torch.log(explainer_unfair.masked_adj.flatten().softmax(-1)),explainer_fair.masked_adj.flatten().softmax(-1)))

            loss-=KL_loss*self.args.KL_weight
            loss.backward(retain_graph=True)

            explainer_fair.optimizer.step()

            mask_density = explainer_fair.mask_density()
            if self.print_training and (epoch%10==0 or epoch+1==self.args.num_epochs):
                print("epoch: ", epoch, "; fidelity_loss: {:.6f}".format(fidelity_loss), "; fidelity_loss_unfair: {:.6f}".format(fidelity_loss_unfair), "; KL : {:.4f}".format(KL_loss.item()), "; WD_fair loss: {:.6f}".format(WD_loss_fair.item()), "; WD_unfair loss: {:.6f}".format(WD_loss_unfair.item()), "; mask_density: {:.2f}".format(mask_density))

        masked_adj_fair = (explainer_fair.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze())
        masked_adj=masked_adj_fair
        masked_adj_unfair = (explainer_unfair.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze())

        self.node_idx.append(node_idx)
        self.explain_true.append(0)

        if np.argmax(self.pred[0][neighbors][node_idx_new])==np.argmax(ypred.cpu().detach().numpy()):
            self.explain_true_wass.append(1)
        else:
            self.explain_true_wass.append(0)
        if np.argmax(self.pred[0][neighbors][node_idx_new])==np.argmax(ypred_unfair.cpu().detach().numpy()):
            self.explain_true_minuswass.append(1)
        else:
            self.explain_true_minuswass.append(0)

        explain_acc = np.array(
            [self.explain_true, self.explain_true_wass, self.explain_true_minuswass, self.wass_dis_ori,
             self.wass_dis, self.wass_dis_unfair,
             self.start_wass_dis_ori,
                                  self.start_wass_dis_ori,self.start_wass_dis,self.start_wass_dis_unfair, self.node_idx])

        fname = 'neighbors_' + 'node_idx_'+str(node_idx)+'.npy'
        with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
            np.save(outfile, np.asarray(neighbors.tolist()+[node_idx_new]))

        fname = 'masked_adj_' + 'node_idx_'+str(node_idx)+'.npy'
        with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
            np.save(outfile, np.asarray([masked_adj.copy(),masked_adj_fair.copy(),masked_adj_unfair.copy()]))

        if self.args.explain_all==True:
            fname = 'explain_result.npy'
            with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
                np.save(outfile, np.asarray(explain_acc.copy()))


        return masked_adj

    def explain_nodes_gnn_stats(self, node_indices):
        for node_idx in tqdm(node_indices):
            self.explain(node_idx)

        explain_acc=np.array([self.explain_true,self.explain_true_wass,self.explain_true_minuswass,self.wass_dis_ori,self.wass_dis,self.wass_dis_unfair,
                                  self.start_wass_dis_ori,self.start_wass_dis,self.start_wass_dis_unfair, self.node_idx])

        if self.args.explain_all==True:
            fname = 'explain_result.npy'
            with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
                np.save(outfile, np.asarray(explain_acc.copy()))

        wass_dis = np.array(explain_acc[4])
        wass_dis_unfair = np.array(explain_acc[5])
        ori_end_dis = np.array(explain_acc[6])

        result=pd.DataFrame([{'dataset':self.args.dataset, 'explainer_backbone':self.args.explainer_backbone, 'GNN_type':self.args.method,
                              'Delta B (Reduced)': '{:.4f}'.format(-np.mean((ori_end_dis - wass_dis) / ori_end_dis) * 1e4),
                              'fair_fidelity': '{:.4f}'.format(np.mean(explain_acc[1])),
                              'Delta B (Promoted)': '{:.4f}'.format(-np.mean((ori_end_dis - wass_dis_unfair) / ori_end_dis) * 1e4),
                              'unfair_fidelity': '{:.4f}'.format(np.mean(explain_acc[2])),
                              }])

        print('Fitting completed.')
        if self.args.explain_all:
            print('Results saved to ' + os.path.join(self.args.logdir, self.folder_name))
        else:
            print(result)

    def extract_neighborhood(self, node_idx):
        neighbors_adj_row = self.neighborhoods[0][node_idx, :]
        node_idx_new = sum(neighbors_adj_row[:node_idx])
        neighbors = np.nonzero(neighbors_adj_row)[0]
        sub_adj = self.adj[0][neighbors][:, neighbors]
        sub_feat = self.feat[0, neighbors]
        sub_label = self.label[0][neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

class ExplainModule(nn.Module):
    def __init__(self, adj, x, model, label, args, use_sigmoid=True):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.args = args
        self.use_sigmoid = use_sigmoid
        num_nodes = adj.size()[1]
        self.PGE_module=nn.Sequential(nn.Linear(self.x.shape[-1], num_nodes),).cuda()
        self.mask = self.construct_edge_mask(num_nodes)

        params = [self.mask]
        if args.explainer_backbone!='GNNExplainer':
            params.extend(self.PGE_module.parameters())

        self.optimizer = util.build_optimizer(args, params)

    def construct_edge_mask(self, num_nodes):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (num_nodes + num_nodes))
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def _masked_adj(self):
        if self.args.explainer_backbone=='PGExplainer':
            sym_mask = self.PGE_module(self.x).squeeze(0)
        else:
            sym_mask = self.mask

        sym_mask = torch.sigmoid(sym_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj

        self.sym_mask=sym_mask
        masked_adj = adj * sym_mask
        return masked_adj

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / (adj_sum+1e-9)

    def forward(self, node_idx):
        x = self.x.cuda() if self.args.gpu else self.x
        self.masked_adj = self._masked_adj()
        ypred, adj_att = self.model(x, self.masked_adj)
        node_pred = ypred[0,node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)
        return res, adj_att
    
    def cal_WD_ypred(self, node_idx,threshold):
        x = self.x.cuda() if self.args.gpu else self.x
        self.masked_adj = self._masked_adj()

        ori_mask = self.sym_mask
        ranking = ori_mask.flatten()
        torch.sort(ranking)

        if threshold < len(ranking):
            threshold_value = ranking[-threshold]
            self.masked_adj = self.adj.cuda() * torch.where(ori_mask>threshold_value, ori_mask, 0)

        new_adj = self.masked_adj

        ypred, adj_att = self.model(x, new_adj)
        node_pred = ypred[0, node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)

        return res, adj_att

    def adj_feat_grad(self, node_idx, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        logit = nn.Softmax(dim=0)(ypred[0, node_idx, :])
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx):
        pred_label_node = pred_label[node_idx]
        logit=pred[pred_label_node]
        pred_loss = -torch.log(logit)
        mask = torch.sigmoid(self.mask)

        size_loss = nn.ReLU()(torch.sum(mask)-self.args.threshold)*self.args.size_weight

        loss = pred_loss + size_loss

        return loss


