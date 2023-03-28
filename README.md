# REFEREE
Open-source code for paper *On Structural Explanation of Biases in Graph Neural Networks*.

## Citation

If you find it useful, please cite our paper. Thank you!

```
@inproceedings{dong2022structural,
  title={On Structural Explanation of Bias in Graph Neural Networks},
  author={Dong, Yushun and Wang, Song and Wang, Yu and Derr, Tyler and Li, Jundong},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={316--326},
  year={2022}
}
```

## Environment
Experiments are carried out on a A6000 GPU with Cuda 11.6.

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  9.39it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  german       GNNExplainer      GAT          -16.3813        0.8800            19.4233          0.8800
```

### 2. An explanation example with PGE-REFEREE

Here we consider PGE-REFEREE as an example to explain a trained GAT model (set dataset='german', method='GAT'). Run
```
python train.py
```
to train the GAT model. Based on a fixed seed 100, we present the sample log as follows.
```
Start fitting ...
100%|█████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:08<00:00, 111.50it/s]
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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:05<00:00,  9.37it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  german        PGExplainer      GAT          -18.9683        0.8800            19.0162          0.9400
```

### 3. Debiasing examples with GE-REFEREE

Here we consider GE-REFEREE as an example to evaluate how it helps with GAT debiasing. First, we train the GAT model by running (set dataset='german', method='GAT', remove=False, num_epoch=100)
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
100%|██████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [10:52<00:00,  1.56it/s]
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
100%|█████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 67.68it/s]
Fitting completed.                                                                                                          
  dataset explainer_backbone GNN_type        SP       EO    Acc  Training Time                                              
0  german       GNNExplainer      GAT  0.141889  0.07013  0.751       1.487375                                              
```
(2) We do configurations in configs.py to operate on 10% nodes (set debias_ratio=10), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...                                                                                                              
100%|█████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 66.75it/s]   
Fitting completed.                                                                                                             
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time                                                
0  german       GNNExplainer      GAT  0.100514  0.044497  0.769       1.511976                                                
```
(3) We do configurations in configs.py to operate on 15% nodes (set debias_ratio=15), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...                                                                                                              
100%|████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 68.45it/s]
Fitting completed.                                                                                                             
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time                                                
0  german       GNNExplainer      GAT  0.078027  0.020239  0.762       1.475075                                                
```
(4) We do configurations in configs.py to operate on 20% nodes (set debias_ratio=20), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...                                                                                                              
100%|████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 67.83it/s]
Fitting completed.                                                                                                             
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time                                                
0  german       GNNExplainer      GAT  0.062272  0.005254  0.736       1.487392                                                
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
Then we do configurations in configs.py to explain all nodes (set dataset='german', method='GAT', explainer_backbone='PGExplainer', explain_all=True), and run 
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
100%|██████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 68.09it/s]
Fitting completed.                                                                                                       
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time                                          
0  german        PGExplainer      GAT  0.165872  0.069233  0.766       1.480103                                          
```
(2) We do configurations in configs.py to operate on 10% nodes (set debias_ratio=10), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...                                                                                                        
100%|██████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 64.23it/s]
Fitting completed.                                                                                                       
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time                                          
0  german        PGExplainer      GAT  0.137354  0.045255  0.772       1.563657                                          
```
(3) We do configurations in configs.py to operate on 15% nodes (set debias_ratio=15), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...                                                                                                           
100%|██████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 69.29it/s]   
Fitting completed.                                                                                                          
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time                                             
0  german        PGExplainer      GAT  0.100468  0.030678  0.761       1.459382                                             
```
(4) We do configurations in configs.py to operate on 20% nodes (set debias_ratio=20), and run
```
python train.py
```
We present a sample log as follows.
```
Start fitting ...                                                                                                           
100%|█████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 63.97it/s]
Fitting completed.                                                                                                          
  dataset explainer_backbone GNN_type        SP        EO    Acc  Training Time                                             
0  german        PGExplainer      GAT  0.062412  0.037029  0.736       1.573705                                             
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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:02<00:00, 18.29it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0    bail       GNNExplainer      GAT          -11.7475        0.8200            13.0241          0.8200
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
Then run (set dataset='bail', method='GAT', explainer_backbone='PGExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:02<00:00, 17.87it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0    bail        PGExplainer      GAT           -8.3560        0.8600            10.4270          0.8800
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
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:07<00:00,  6.39it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  credit       GNNExplainer      GAT          -10.4154        0.8600            16.9179          0.8800
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
Then run (set dataset='credit', method='GAT', explainer_backbone='PGExplainer')
```
python explainer_main.py
```
We have the sample log as follows.
```
Start fitting ...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:07<00:00,  6.48it/s]
Fitting completed.
  dataset explainer_backbone GNN_type Delta B (Reduced) fair_fidelity Delta B (Promoted) unfair_fidelity
0  credit        PGExplainer      GAT          -14.7684        0.9000            14.1093          0.9000
```
