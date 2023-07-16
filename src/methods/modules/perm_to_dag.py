import logging
import warnings
from multiprocessing import Pool

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.stats import norm
from src.methods.modules.LCIT import UniformFlow
from src.utils import strip_outliers
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from pygam import GAM
from pygam.terms import s, TermList

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig()
logger = logging.getLogger(__name__)

EPS = 1e-6

class MultipleCNF(LightningModule):
    def __init__(self, dz, n_components, hidden_sizes, lr, l1, l2, verbose):
        super().__init__()
        self.save_hyperparameters()
        self.validation_outputs = []

        self.cnfs = nn.ModuleDict()
        for i in range(1, dz):
            for j in range(i):
                self.cnfs[f'{j} {i}'] = nn.ModuleList([
                    UniformFlow(dz=i - 1, n_components=n_components, hidden_sizes=hidden_sizes),
                    UniformFlow(dz=i - 1, n_components=n_components, hidden_sizes=hidden_sizes)
                ])

    def loss(self, X):
        loss = []
        for ji, (x_cnf, y_cnf) in self.cnfs.items():
            j, i = map(int, ji.split())
            K = np.delete(np.arange(i), j)
            Z = X[:, K]
            ex, log_px = x_cnf(X[:, j], Z)
            ey, log_py = y_cnf(X[:, i], Z)
            loss.append(log_px + log_py)

        loss = -torch.mean(torch.column_stack(loss))
        l1 = sum(torch.sum(torch.abs(p)) for p in self.parameters() if p.requires_grad_)
        return loss + self.hparams.l1 * l1

    def training_step(self, batch, batch_idx):
        X, = batch
        loss = self.loss(X)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, = batch
        loss = self.loss(X)
        self.validation_outputs.append(loss)

        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.validation_outputs.clear()
        self.log('val_loss', avg_loss)
        
    def transform(self, X):
        self.eval()
        res = {}
        for ji, (x_cnf, y_cnf) in self.cnfs.items():
            j, i = map(int, ji.split())
            K = np.delete(np.arange(i), j).tolist()
            Z = X[:, K]
            ex, log_px = x_cnf(X[:, j], Z)
            ey, log_py = y_cnf(X[:, i], Z)

            ex = ex.detach().cpu().numpy()
            ey = ey.detach().cpu().numpy()
            res[(j, i)] = (ex, ey)

        return res

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.l2)
        scheduler = ReduceLROnPlateau(optimizer, factor=.1, patience=3, verbose=self.hparams.verbose)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def fit(self, X, batch_size, max_epochs, verbose, gpus=None, callbacks=None):
        gpus = gpus or 0
        callbacks = callbacks or []
        N, _ = X.shape
        train_size = int(N * 0.8)
        valid_size = N - train_size
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train, valid = random_split(TensorDataset(X), lengths=[train_size, valid_size])
            train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(valid, batch_size=batch_size)
            early_stopping = EarlyStopping(
                mode='min',
                monitor='val_loss',
                patience=10,
                verbose=verbose
            )
            callbacks.append(early_stopping)
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.device_count() else "cpu",
                max_epochs=max_epochs,
                logger=verbose,
                enable_checkpointing=verbose,
                enable_progress_bar=verbose,
                enable_model_summary=verbose,
                deterministic=True,
                callbacks=callbacks,
                detect_anomaly=verbose,
                gradient_clip_val=1,
                gradient_clip_algorithm="value"
            )
            trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            logs, = trainer.validate(model=self, dataloaders=val_dataloader, verbose=verbose)
        
        return logs

def process(task):
    task_id, (dataset, (ci_test, ci_params)) = task
    res = dataset.copy()
    X = res.pop('X').copy()
    Y = res.pop('Y').copy()
    Z = res.pop('Z').copy()

    try:
        res['p_value'] = ci_test(X=X, Y=Y, Z=Z, **ci_params)
    except Exception as e:
        print(e)
        print('Error', res)
    
    return res

def CIT_perm2dag_sequential(X, perm, ci_test, alpha, n_jobs=1, verbose=False, **kwargs):
    N, d = X.shape

    datasets = []
    for i in range(d):
        for j in range(i):
            K = [perm[k] for k in range(i) if k != j]
            x = X[:, perm[j]]
            y = X[:, perm[i]]
            z = X[:, K]
            datasets.append((dict(i=perm[i], j=perm[j], X=x, Y=y, Z=z), (ci_test, kwargs)))
    
    tasks = list(enumerate(datasets))
    if n_jobs > 1:
        with Pool(n_jobs) as p:
            if verbose:
                res = list(tqdm(p.imap_unordered(process, tasks, chunksize=1), total=len(tasks)))
            else:
                res = list(p.imap_unordered(process, tasks, chunksize=1))
    else:
        if verbose:
            res = list(map(process, tqdm(tasks)))
        else:
            res = list(map(process, tasks))

    A = np.zeros((d, d))
    for r in res:
        if verbose:
            print(f'{r["j"]} -> {r["i"]}: {r["p_value"]}')
        if r['p_value'] <= alpha:
            A[r['j'], r['i']] = 1

    return A

def CIT_perm2dag_parallel(X, perm, normalize=True, hidden_sizes=4, n_components=4, lr=0.1, l1=1e-5, l2=5e-5, batch_size=512, max_epochs=100, random_state=0, verbose=False, **kwargs):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    if random_state is not None:
        torch.random.fork_rng(enabled=True)
        torch.random.manual_seed(random_state)
        
    N, d = X.shape
    inv_perm = np.zeros(d).astype(int)
    inv_perm[perm] = np.arange(d)
    X = X[:, perm]
    if normalize:
        X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
        X = strip_outliers(X, 0.025)
    X = torch.tensor(X).float()

    model = MultipleCNF(d, hidden_sizes=hidden_sizes, n_components=n_components, lr=lr, l1=l1, l2=l2, verbose=verbose)
    model.fit(X, batch_size=batch_size, max_epochs=max_epochs, verbose=verbose, **kwargs)
    
    J, I, Ex, Ey = [], [], [], []
    for (j, i), (e_x, e_y) in model.transform(X).items():
        J.append(j)
        I.append(i)
        Ex.append(e_x)
        Ey.append(e_y)
    
    Ex, Ey = map(np.column_stack, (Ex, Ey))
    Ex, Ey = map(lambda x: np.clip(x, EPS, 1 - EPS), (Ex, Ey))
    Ex, Ey = map(norm.ppf, (Ex, Ey))
    Ex, Ey = map(lambda x: (x - np.mean(x, axis=0)) / np.std(x, axis=0), (Ex, Ey))
    r = np.mean(Ex * Ey, axis=0)
    r = np.clip(r, -1 + EPS, 1 - EPS)
    stat = 0.5 * np.sqrt(N - 3) * np.log1p(2 * r / (1 - r))
    p_value = 2 * (1 - norm.cdf(abs(stat)))

    A = np.ones((d, d))
    A[J, I] = p_value

    A = A[np.ix_(inv_perm, inv_perm)]
    return A

def train_gam(X, y, n_basis=10):
    N, d = X.shape
    if N / d < 3 * n_basis:
        n_basis = int(np.ceil(N / (3 * d)))
    mgcv_formula = [s(a, n_splines=n_basis) for a in range(d)]
    gam = GAM(terms=TermList(*mgcv_formula)).fit(X=X, y=y)
    return gam

def cam_pruning_perm2dag(X, perm, verbose=False, **kwargs):
    N, d = X.shape
    A = np.ones((d, d))
    for i in range(1, d):
        gam = train_gam(X[:, perm[:i]], X[:, perm[i]])
        p_values = gam.statistics_['p_values']
        A[perm[:i], perm[i]] = p_values[:-1]
    return A