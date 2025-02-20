## Requirements overview

Our implementation relies on the [BackPACK](https://github.com/f-dangel/backpack/) package in [PyTorch](https://pytorch.org/) to easily compute gradient variances.

- python == 3.8.19
- torch == 1.12.0+cu113
- torchvision == 0.13.0+cu113
- backpack-for-pytorch == 1.3.0
- numpy == 1.24.4

## Procedure

Install this repository and the dependencies using pip:
```bash
$ conda create --name cmnist python=3.8.19
$ conda activate cmnist
$ cd ColoredMNIST
$ pip install -r requirements.txt
```

## Colored MNIST in the IRM setup

This github enables the replication of our main experiments on Colored MNIST in the setup defined by [IRM](https://github.com/facebookresearch/InvariantRiskMinimization/tree/master/code/colored_mnist)

### Main results

To reproduce the results from our paper:

```bash
python train_coloredmnist.py \
  --algorithm "$algorithm" \
  --n_restarts 5 \
  --ssl_weight $ssl_weight \
  --ssl_temp 0.7 \
  --proj_mask $proj_mask \
  --output_path experiment_results.csv
```

 where `algorithm` is either:
- ```erm``` for Empirical Risk Minimization
- ```irm``` for [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893)
- ```rex``` for [Out-of-Distribution Generalization via Risk Extrapolation](https://icml.cc/virtual/2021/oral/9186)
- ```fishr``` for [Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization](https://arxiv.org/abs/2109.02934))

Set `ssl_weight` for the result without our `DivIL` as the baseline.

Results will be printed at the end of the script, averaged over 10 runs, also will be reported to `.csv` file set by `output_path`. Note that all hyperparameters are taken from the seminal [IRM](https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/reproduce_paper_results.sh) implementation.