## Installation and data preparation

Our code is based on the following libraries:

```
torch==1.12.1
torch-geometric==2.5.0
```

plus the [DrugOOD](https://github.com/tencent-ailab/DrugOOD) benchmark repo.

The data used in the paper can be obtained following these [instructions](./dataset_gen/README.md).

## Code Structure

We changed the code based on [CIGA](https://github.com/LFhase/CIGA/tree/main).
`main-batch_aug.py` is our main Python file, 

1) batch means that we do contrastive learning in every batch graph, 
2) aug means we add self-supervised learning based on the random augmented graph, we implement edge removing, node dropping and subgraph extraction based on [GraphCL](https://github.com/Shen-Lab/GraphCL).

`main-complement.py` is the completion of the `main-batch_aug.py`, we just split it to shorten the code length. It contains the parameters, evaluator, and model initial.

Also, we implement wandb to visualize the loss, erank, accuracy, and other metrics, you can re-use it in your wandb account or comment out them.


## Reproduce results
### Main results
We provide the hyperparameter tuning and evaluation details in the paper and appendix.
In the below, we give a brief introduction of the commands and their usage in our code. 

To obtain results of ERM, simply run 

```
python main.py --erm
```

with corresponding datasets and model specifications.

Running with our DivIL:

- `--ginv_opt` specifies the interpretable GNN architectures, which can be `asap` or `gib` to test with ASAP or GIB respectively.
- `--r` is used in interpretable GNN architectures to define the interpretable ratio, which corresponds to the size of $G_c$. In our implementation, since the entire graph is utilized, the default value is set to `-1`.
- `--c_rep` controls the inputs of the contrastive learning, e.g., the graph representations from the featurizer or from the classifier
- `--c_in` controls the inputs to the classifier, e.g., the original graph features or the features from the featurizer
- `--s_rep` controls the inputs for maximizing $I(\hat{G_s};Y)$, e.g., the graph representation of $\hat{G_s}$ from the classifier or the featurizer.
- `--ssl` is used to set the weight for self-supervised contrastive learning.
- `--type2` defines the type of graph augmentation and `--p2` specifies the augmentation ratio. 
- `--ssl_d` specifies the remaining dimensionality percentage for the augmented graphs ($g_1$ and $g_2$).

For more hyperparameter settings, please refer to `main-complement.py`.

Running with the baselines:

- To test with DIR, simply specify `--ginv_opt` as default and `--dir` a value larger than `0`.
- To test with invariant learning baselines, specify `--num_envs=2` and
  use `--irm_opt` to be `irm`, `vrex`, `eiil` or `ib-irm` to specify the methods,
  and `--irm_p` to specify the penalty weights.

Due to the additional dependence of an ERM reference model in CNC, we need to train an ERM model and save it first,
and then load the model to generate ERM predictions for positive/negative pairs sampling in CNC. 
Here is a simplistic example:
```
python main.py --erm --contrast 0 --save_model
python main.py --erm --contrast 1  -c_sam 'cnc'
```