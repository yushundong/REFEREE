# REFEREE
Open-source code for paper *On Structural Explanation of Biases in Graph Neural Networks*.

## Citation

If you find it useful, please cite our paper. Thank you!

```
@inproceedings{dong2022referee,
  title={On Structural Explanation of Biases in Graph Neural Networks},
  author={Dong, Yushun and Wang, Song and Wang, Yu and Derr, Tyler and Li, Jundong},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  year={2022}
}
```

## Environment
Experiments are carried out on a Tesla V100 GPU with Cuda 11.6.

Dependencies can be found in requirements.txt.

Notice: Cuda is enabled for default settings.

## Usage
We have three datasets for experiments. To choose the dataset/explainer backbone/GNN models, do configurations in configs.py. Then run the following command to train the GNN to be explained:
```
python train.py
```

To explain the trained GNN, first do configurations in configs.py. Then run the following command to explain the GNN:
```
python explainer_main.py
```





## Log examples on German

### 1. An explanation example with GE-REFEREE


Here we consider GE-REFEREE as an example to explain a trained GAT model (set dataset='german', method='GAT'). Run
```
python train.py
```
to train the GAT model. Based on a fixed seed 100, we present the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:08<00:00, 111.50it/s]
Fitting completed.
  dataset GNN_type    Acc  Training Time
0  german      GAT  0.712       9.004312
```
Then run (set dataset='german', method='GAT', explainer_backbone='GNNExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:37<00:00,  1.32it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  german       GNNExplainer      GAT          -19.8872        0.8200            18.2604          0.9400
```

### 2. An explanation example with PGE-REFEREE

Here we consider PGE-REFEREE as an example to explain a trained GAT model (set dataset='german', method='GAT'). Run
```
python train.py
```
to train the GAT model. Based on a fixed seed 100, we present the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:09<00:00, 105.68it/s]
Fitting completed.
  dataset GNN_type    Acc  Training Time
0  german      GAT  0.712       9.489083
```
Then run (set dataset='german', method='GAT', explainer_backbone='PGExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:37<00:00,  1.35it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  german        PGExplainer      GAT          -15.6395        0.8600            18.1380          0.9200
```

### 3. Debiasing examples with GE-REFEREE

Here we consider GE-REFEREE as an example to evaluate how it helps with GAT debiasing. First, we train the GAT model by running (set dataset='german', method='GAT', remove=True)
```
python train.py
```
Based on a fixed seed 100, we present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 64.25it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO   Acc  Training Time
0  german       GNNExplainer      GAT  0.201403  0.129762  0.77        1.56883
```
Then we do configurations in configs.py to explain all nodes (set dataset='german', method='GAT', explainer_backbone='GNNExplainer', explain_all=True), and run 
```
python explainer_main.py
```
We present a sample log as follows.
```
Start fitting ...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [11:38<00:00,  1.43it/s]
Fitting completed.
Results saved to log/german_GNNExplainer_GAT
```
Finally, (1) we do configurations in configs.py to operate on 5% nodes (set debias_ratio=5), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 66.33it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time
0  german       GNNExplainer      GAT  0.191819  0.071307  0.774       1.519318
```
(2) We do configurations in configs.py to operate on 10% nodes (set debias_ratio=10), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 66.11it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time
0  german       GNNExplainer      GAT  0.117999  0.065913  0.721       1.516468
```
(3) We do configurations in configs.py to operate on 15% nodes (set debias_ratio=15), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 66.47it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time
0  german       GNNExplainer      GAT  0.069705  0.016371  0.745       1.511232
```
(4) We do configurations in configs.py to operate on 20% nodes (set debias_ratio=20), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 65.60it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time
0  german       GNNExplainer      GAT  0.040206  0.007049  0.738       1.530935
```


### 4. Debiasing examples with PGE-REFEREE


Here we consider PGE-REFEREE as an example to evaluate how it helps with GAT debiasing. First, we train the GAT model by running (set dataset='german', method='GAT', remove=True)
```
python train.py
```
Based on a fixed seed 100, we present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 59.33it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO   Acc  Training Time
0  german        PGExplainer      GAT  0.201403  0.129762  0.77       1.693708
```
Then we do configurations in configs.py to explain all nodes (set dataset='german', method='GAT', explainer_backbone='GNNExplainer', explain_all=True), and run 
```
python explainer_main.py
```
We present a sample log as follows.
```
Start fitting ...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [11:10<00:00,  1.49it/s]
Fitting completed.
Results saved to log/german_PGExplainer_GAT
```
Finally, (1) we do configurations in configs.py to operate on 5% nodes (set debias_ratio=5), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 64.42it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time
0  german        PGExplainer      GAT  0.161057  0.059283  0.744       1.563453
```
(2) We do configurations in configs.py to operate on 10% nodes (set debias_ratio=10), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 64.59it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP      EO    Acc  Training Time
0  german        PGExplainer      GAT  0.148901  0.0637  0.721       1.560527
```
(3) We do configurations in configs.py to operate on 15% nodes (set debias_ratio=15), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 65.90it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time
0  german        PGExplainer      GAT  0.110706  0.036959  0.732       1.529081
```
(4) We do configurations in configs.py to operate on 20% nodes (set debias_ratio=20), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 59.93it/s]
Fitting completed.
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time
0  german        PGExplainer      GAT  0.056709  0.003669  0.728       1.680408
```




## Log examples on Bail

### 1. An explanation example with GE-REFEREE


Here we consider GE-REFEREE as an example to explain a trained GAT model (set dataset='bail', method='GAT'). Run
```
python train.py
```
to train the GAT model. Based on a fixed seed 100, we present the sample log as follows.
```
Start fitting ...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:36<00:00, 10.32it/s]
Fitting completed.
  dataset GNN_type       Acc  Training Time
0    bail      GAT  0.958519     105.481607
```
Then run (set dataset='bail', method='GAT', explainer_backbone='PGExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:44<00:00,  1.13it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0    bail       GNNExplainer      GAT          -10.5899        0.8800            20.2563          0.8800
```

### 2. An explanation example with PGE-REFEREE

Here we consider PGE-REFEREE as an example to explain a trained GAT model (set dataset='bail', method='GAT'). Run
```
python train.py
```
to train the GAT model. Based on a fixed seed 100, we present the sample log as follows.
```
Start fitting ...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [01:37<00:00, 10.27it/s]
Fitting completed.
  dataset GNN_type       Acc  Training Time
0    bail      GAT  0.958519     105.194559
```
Then run (set dataset='bail', method='GAT', explainer_backbone='GNNExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:42<00:00,  1.17it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0    bail        PGExplainer      GAT          -10.5172        0.9600             8.7523          0.8200
```







## Log examples on Credit

### 1. An explanation example with GE-REFEREE


Here we consider GE-REFEREE as an example to explain a trained GAT model. Run (set dataset='credit', method='GAT')
```
python train.py
```
to train the GAT model. Based on a fixed seed 100, we present the sample log as follows.
```
Start fitting ...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:31<00:00,  4.74it/s]
Fitting completed.
  dataset GNN_type       Acc  Training Time
0  credit      GAT  0.802333      228.60836
```
Then run (set dataset='credit', method='GAT', explainer_backbone='GNNExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:28<00:00,  1.77it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  credit       GNNExplainer      GAT          -10.8445        0.8800            17.7664          0.8600
```

### 2. An explanation example with PGE-REFEREE

Here we consider PGE-REFEREE as an example to explain a trained GAT model. Run (set dataset='credit', method='GAT')
```
python train.py
```
to train the GAT model. Based on a fixed seed 100, we present the sample log as follows.
```
Start fitting ...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:31<00:00,  4.73it/s]
Fitting completed.
  dataset GNN_type       Acc  Training Time
0  credit      GAT  0.802333      230.23522
```
Then run (set dataset='credit', method='GAT', explainer_backbone='GNNExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:28<00:00,  1.74it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  credit        PGExplainer      GAT          -11.1100        0.8800            12.6806          0.8800
```
