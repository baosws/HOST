import sys

sys.path.append('')
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool

import pandas as pd, numpy as np
from src.data_gen import simulate_cd
from src.methods import methods_dict
from src.utils import Timer, read_config, str_to_matrix
from src.methods.modules.perm_to_dag import CIT_perm2dag_parallel, cam_pruning_perm2dag
from tqdm import tqdm
from pathlib import Path

def process(task):
    task_id, (dataset, ci_params) = task
    res = dataset.copy()
    X = res.pop('data').copy()

    try:
        with Timer() as t:
            res['p-values'] = CIT_perm2dag_parallel(X, perm=res['Perm'], **ci_params)
            # res['p-values'] = cam_pruning_perm2dag(X, perm=res['Perm'], **ci_params)
        res['Time'] = t.elapsed
        assert isinstance(res['p-values'], np.ndarray)
    except Exception as e:
        print('Error', res['Method'], res['N'], res['d'], res['dag_type'], res['k'], res['random_state'])
        print(e)
    
    return res

if __name__ == '__main__':
    # CONFIG------------------------------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', default=['OURS'])
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    methods = args.methods

    EXP_NAME = Path(__file__).absolute().parent.name
    config_path = f'experiments/{EXP_NAME}/configs.yml'
    configs = read_config(config_path)
    method_params = configs.get('methods', {})
    for method in methods:
        method_params[method] = method_params.get(method, {})
        print(f'{method}: {method_params[method]}')

    ci_params = configs['ci_params']
    print(f'{ci_params = }')

    # DATA GEN----------------------------------------------------------
    path = f'experiments/{EXP_NAME}/results/result_{"-".join(sorted(methods))}.csv'
    df = pd.read_csv(path)
    df = df[df.random_state < 5]
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['Perm'] = df['Perm'].apply(lambda x: list(map(int, x.split())))
    datasets = df.to_dict('records')
    for dataset in datasets:
        X, GT = simulate_cd(**dataset)
        assert (GT == dataset['GT']).all()
        dataset['data'] = X

    # RUN---------------------------------------------------------------
    tasks = list(enumerate([(dataset, ci_params) for dataset in datasets]))
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as p:
            res = list(tqdm(p.imap_unordered(process, tasks, chunksize=1), total=len(tasks)))
    else:
        res = list(map(process, tqdm(tasks)))

    # VISUALIZE---------------------------------------------------------
    df = pd.DataFrame(res)
    df['GT'] = df['GT'].apply(lambda x: ' '.join(map(str, x.ravel())))
    df['Perm'] = df['Perm'].apply(lambda x: ' '.join(map(str, x)))
    df['p-values'] = df['p-values'].apply(lambda x: ' '.join(map(str, x.ravel() if isinstance(x, np.ndarray) else x)))
    path = f'experiments/{EXP_NAME}/results/result_dag_{"-".join(sorted(methods))}.csv'
    df.to_csv(path, index=False)