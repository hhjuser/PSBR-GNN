# Edge-Private Graph Neural Networks via Aggregation-Preserving Graph Reconstruction  (PSBR-GNN)

## Requirements
This artifact relies on the environment setup defined in `requirements.txt`.


## File structure
Our artifact consists of the source code, organized with the following file structure:
- `./dataset`: Stores all datasets used in the study. 
- `./baselines`: Contains privacy-enhanced variants of existing methods.
- `gcn.py`: Implements multi-layer neighborhood aggregation for GCN and configures training parameters.
- `utils.py`: Implements aggregation-preserving private graph reconstruction for GNNs.
- `train.py`: Implements the training process for the model and generates the main experimental results.
- `utils_bound_error.py`: Implements theoretical bound and empirical error computation for SBR-GNN and PSBR-GNN.
- `run_bound_error.py`: Runs theoretical bound and empirical error evaluation for SBR-GNN and PSBR-GNN.

## Running
If you want to individually train and evaluate the models on any of the datasets mentioned in the paper, run the following command:  

```
dataset arguments:
  --dataset        <str>       name of the dataset ( "cora", "citeseer", "facebook", "DBLP", "CS", "Physics")
  --val_rate       <float>     fraction of nodes used for validation (default: 0.75)

training arguments:
  --epochs           <int>        maximum number of training epochs (default: 200)
  --lr               <float>      learning rate (default: 0.01)
  --weight-decay     <float>      weight decay (default: 0.0005)
  --run_times        <int>        runtimes (default: 10)

model arguments:
  --hidden                <int>      dimension of the hidden layers (default: 64)
  --dropout               <float>    dropout rate (between zero and one) (default: 0.5)
  --communities           <float>    grouping ratio r (default: 0-1)
  --sigma_degree          <float>     the reweighting parameter  (default: 0-1)
  --alpha                 <float>     the feature similarity weight  (default: 0-1)

privacy arguments:
  --delta       <float>        privacy budget allocation parameter (default: 0.1)
  --epsilon     <float>        total privacy budget  (default: 1-8)

``` 
