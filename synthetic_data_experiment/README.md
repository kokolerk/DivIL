## Synthetic Data Experiment

This part aims to conduct simple experiments using synthetic data. 

In this repository, we use [DomainBed](https://github.com/facebookresearch/DomainBed) as our base invariance learning algorithm implementation for the synthetic data experiment. DomainBed is a Pytorch suite containing benchmark datasets and algorithms for domain generalization and transfer learning research.


### How to Run

To reproduce the results from our paper:

1. Install dependencies:
    ```sh
    $ pip install -r requirements.txt
    ```

2. Run the main program:
    ```sh
    $ python main.py
    ```

### Modifications to DomainBed

In our experiments, we simply use `infoNCE` as our unsupervised contrastive learning (UCL) method.

In order to validate the effectiveness of unsupervised contrastive learning in alleviating the over-invariance phenomenon, we made simple modifications to the DomainBed codebase. Specifically, we added the `use_ucl` parameter to some of the algorithms in `domainbed_utils/algorithms.py`. This parameter allows us to switch between the original algorithm and the algorithm with unsupervised contrastive learning. If `use_ucl` is set to `True`, the `infoNCE` loss will be calculated and added to the original loss.

Currently, we have only added the `use_ucl` parameter to `IRM` and `VREx`.
