import sys

sys.path.append('')
from argparse import ArgumentParser
from itertools import product
from multiprocessing import Pool

import pandas as pd
from experiments.synthetic.viz import viz_perm_by_N
from src.data_gen import simulate_cd
from src.methods import methods_dict
from src.utils import Timer, read_config
from tqdm import tqdm
from pathlib import Path

def process(task):
    task_id, (dataset, (func, params)) = task
    res = dataset.copy()
    res['Method'] = func.__name__
    X = res.pop('data').copy()

    try:
        with Timer() as t:
            res['Perm'] = func(X=X, **params)
        res['Time'] = t.elapsed
    except Exception as e:
        print('Error', res)
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

    data_config = configs['data']
    print(f'{data_config = }')
    T = data_config['T']
    
    # DATA GEN----------------------------------------------------------
    datasets = []
    res = []
    for dag_type in data_config['dag_type']:
        for k in data_config['k']:
            for i in range(T):
                for N in data_config['N']:
                    for d in data_config['d']:
                        X, GT = simulate_cd(N=N, d=d, k=k, dag_type=dag_type, random_state=i)
                        datasets.append(dict(
                            data=X,
                            dag_type=dag_type,
                            k=k,
                            d=d,
                            N=N,
                            GT=GT,
                            random_state=i
                        ))

    # RUN---------------------------------------------------------------
    tasks = list(enumerate(product(datasets, [(methods_dict[method], method_params[method]) for method in methods])))
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as p:
            res = list(tqdm(p.imap_unordered(process, tasks, chunksize=1), total=len(tasks)))
    else:
        res = list(map(process, tqdm(tasks)))

    # VISUALIZE---------------------------------------------------------
    df = pd.DataFrame(res)
    df['GT'] = df['GT'].apply(lambda x: ' '.join(map(str, x.ravel())))
    df['Perm'] = df['Perm'].apply(lambda x: ' '.join(map(str, x)))
    path = f'experiments/{EXP_NAME}/results/result_{"-".join(sorted(methods))}.csv'
    df.to_csv(path, index=False)
    g = viz_perm_by_N(path)