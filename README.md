# Heteroscedastic Causal Structure Learning (HOST)


<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.8-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/abs/2307.07973"><img src="https://img.shields.io/badge/arXiv-2307.07973-b31b1b.svg" alt="arXiv" height="18"></a>
</p>

<p align="center">
  <a href="#installation">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#citation">Citation</a>
</p>

This is the implementation of our paper: Bao Duong and Thin Nguyen. [Heteroscedastic Causal Structure Learning](https://arxiv.org/abs/2307.07973). Accepted at the 26th European Conference on Artificial Intelligence (ECAI 2023).

## Setup

```bash
conda env create -n host --file environment.yml
conda activate host
```

## Usage

```python
from src.data_gen import simulate_cd
from src.methods.HOST import HOST

if __name__ == '__main__':
    data, dag_gt = simulate_cd(N=500, d=5, dag_type='ER', k=1, random_state=0)
    perm, dag = HOST(X=data, return_dag=True)
```

## Experiments

For example, to run Figure 1 experiment:
```
python experiments/synthetic/perm.py --methods HOST --n_jobs=8
```

Experiment configuration can be set in `experiments/*/config.yml`, and result dataframes are stored in `experiments/*/results/` after the command is finished.

## Citation

```
@article{duong2023heteroscedastic,
      title={Heteroscedastic Causal Structure Learning}, 
      author={Bao Duong and Thin Nguyen},
      year={2023},
      journal={arXiv preprint arXiv:2307.07973}
}
```
