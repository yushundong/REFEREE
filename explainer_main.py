import argparse
import os
import json
import sklearn.metrics as metrics
import pandas as pd
from tensorboardX import SummaryWriter
import pickle
import shutil
import torch
import numpy as np
import models
import util
import explain
import scipy.sparse as sp
import networkx as nx
import warnings
import random
import configs
warnings.filterwarnings("ignore")

def main():

    prog_args = configs.arg_parse_explain()

    # Based on a set of re-finetuned hyper-parameters for more stable performance
    configs_loaded=json.load(open('./configs/configs.json'))['{}_{}_GAT'.format(prog_args.dataset,prog_args.explainer_backbone)]
    prog_args.fidelity_unfair_weight = configs_loaded['fidelity_unfair_weight']
    prog_args.fidelity_fair_weight = configs_loaded['fidelity_fair_weight']
    prog_args.WD_fair_weight = configs_loaded['WD_fair_weight']
    prog_args.WD_unfair_weight = configs_loaded['WD_unfair_weight']
    prog_args.KL_weight = configs_loaded['KL_weight']

    filename='./ckpt/'+prog_args.dataset+'_'+prog_args.method+'.pth.tar'
    ckpt = torch.load(filename)

    cg_dict = ckpt["cg"]
    input_dim = cg_dict["feat"].shape[2]
    num_classes = cg_dict["pred"].shape[2]

    model = models.GNN(input_dim=input_dim, hidden_dim=prog_args.hidden_dim, embedding_dim=prog_args.output_dim, label_dim=num_classes, num_layers=prog_args.num_gc_layers, bn=prog_args.bn, args=prog_args,)
    if prog_args.gpu:
        model = model.cuda()
    model.load_state_dict(ckpt["model_state"])

    explainer = explain.REFEREE(model=model, adj=cg_dict["adj"], feat=cg_dict["feat"], label=cg_dict["label"], pred=cg_dict["pred"], train_idx=cg_dict["train_idx"], args=prog_args)

    if prog_args.explain_all:
        node_list=explainer.test_idx
    else:
        node_list=json.load(open('./configs/configs.json'))['node_list_{}_{}_GAT'.format(prog_args.dataset, prog_args.explainer_backbone)][prog_args.group_sample]

    print('Start fitting ...')
    explainer.explain_nodes_gnn_stats(node_list)

if __name__ == "__main__":
    main()

