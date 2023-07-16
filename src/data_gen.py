import networkx as nx
import numpy as np


def rand_coef(rng):
    return rng.uniform(0.2, 0.5) * (rng.randint(2) * 2 - 1)

def linear(X, rng):
    X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    return X * rand_coef(rng) + rand_coef(rng)

def square(X, rng):
    X = (X - np.mean(X)) / np.std(X)
    return X ** 2

def cube(X, rng=None):
    X = (X - np.mean(X)) / np.std(X)
    return X ** 3

def exp(X, rng=None):
    return np.power(1.5, X)

def sin(X, rng=None):
    return np.sin(X * 2 * np.pi)

def cos(X, rng=None):
    return np.cos(X * 2 * np.pi)

def inverse(X, rng=None):
    return 1 / (X - np.min(X) + 1)

def nexp(X, rng=None):
    return np.exp(-X)

def log(X, rng=None):
    X = (X - np.mean(X)) / np.std(X)
    return np.log1p(X - np.min(X))
    
def sigmoid(X, rng=None):
    return 1 / (1 + np.exp(-X))

EPS = 1e-10

def simulate_dag(d, dag_type, rho=None, k=None, max_parents=None, rng=None, random_state=None):
    assert dag_type in ['ER', 'SF', 'Unif']
    if rng is None:
        rng = np.random.RandomState(random_state)
    if dag_type == 'ER':
        U = np.triu(np.ones((d, d)), 1)
        if rho is None:
            assert k is not None
            rho = 2 * k / (d - 1)
        x, y = np.nonzero(U)
        A = np.zeros((d, d))
        idx = rng.choice(np.arange(len(x)), size=k * d, replace=False)
        A[x[idx], y[idx]] = 1
    elif dag_type == 'SF':
        if k is None:
            assert rho is not None
            k = int(rho * (d - 1))
        G = nx.barabasi_albert_graph(n=d, m=k, seed=rng)
        A = nx.adjacency_matrix(G).toarray()
        A = np.triu(A, 1)
    elif dag_type == 'Unif':
        assert max_parents is not None
        A = np.zeros((d, d))
        for i in range(1, d):
            n_parents = rng.randint(0, min(i, max_parents) + 1)
            parents = rng.choice(np.arange(i), size=n_parents, replace=False)
            A[parents, i] = 1

    return A.astype(int)

def simulate_cd(N, d, dag_type, linear_params=True, rho=None, k=None, max_parents=None, random_state=None, random_permute=True, verbose=False, **kwargs):
    rng = np.random.RandomState(random_state)

    A = simulate_dag(d, dag_type, rho=rho, k=k, max_parents=max_parents, rng=rng)
    D = np.zeros((N, d))
    noises = np.zeros((N, 0))
    if linear_params:
        funcs = [linear]
    else:
        funcs = [linear, sin, square, sigmoid, log]
    
    for i in range(d):
        parents, = np.nonzero(A[:, i])
        fe1 = funcs[rng.randint(len(funcs))]
        fe2 = funcs[rng.randint(len(funcs))]
        e1 = fe1(D[:, parents], rng)
        e2 = fe2(D[:, parents], rng)
        logs = [parents.tolist(), fe1.__name__, fe2.__name__]
        if verbose:
            print(f'Parents[{i}]: {logs}')

        if len(parents):
            e1, e2 = map(lambda x: np.sum(x, axis=1), (e1, e2))
            e1, e2 = map(lambda x: (x - x.min()) / (x.max() - x.min()) * 8 - 4, (e1, e2)) # avoid too extreme variances
        else:
            e1 = np.zeros(N)
            e2 = np.zeros(N)

        eta1 = e2
        eta2 = -np.exp(e1)

        t = -eta1 / (2 * eta2)
        s = np.log(-1 / (2 * eta2)) / 2
        
        assert np.isnan(s).any() == False
        assert np.isnan(t).any() == False
        noise = rng.randn(N)
        D[:, i] = noise * np.exp(s) + t
        noises = np.column_stack((noises, noise))
    assert np.isnan(D).any() == False
    
    noises = np.column_stack(noises)
    if random_permute:
        order = rng.permutation(d)
        A = A[np.ix_(order, order)]
        D = D[:, order]

    return D, A

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    N = 10000
    d = 2
    random_state = 793
    X, GT = simulate_cd(N=N, d=d, linear_only=False, dag_type='SF', k=1, random_state=random_state, return_noises=True, random_permute=False, verbose=True)
    df = pd.DataFrame(X)
    sns.pairplot(df, diag_kws={'bins': 100}, grid_kws={'sharex': False, 'sharey': False})
    plt.savefig('data.png')
