import logging
import warnings

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.methods.modules.perm_to_dag import CIT_perm2dag_parallel
from src.methods.modules.AffineFlow import AffineFlow
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.stats import shapiro

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.basicConfig()
logger = logging.getLogger(__name__)

EPS = 1e-6
class MultipleCNF(LightningModule):
    def __init__(self, d_rem, d_cond, hidden_sizes, lr, l1, l2, verbose):
        super().__init__()
        self.save_hyperparameters()
        self.validation_outputs = []

        self.cnf = AffineFlow(d_in=d_cond, d_out=d_rem, dh=hidden_sizes)

    def objective(self, X, Z):
        e, ll, s, t = self.cnf(X, Z)

        ell = torch.mean(ll)
        norm = sum(torch.norm(p, 1) for p in self.parameters() if p.requires_grad and p.numel())
        return -ell + self.hparams.l1 * norm

    def training_step(self, batch, batch_idx):
        X, Z = batch
        loss = self.objective(X, Z)
        self.log('train_obj', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, Z = batch
        loss = self.objective(X, Z)
        self.validation_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_outputs).mean()
        self.validation_outputs.clear()
        self.log('val_obj', avg_loss)

    def transform(self, X, Z):
        self.eval()
        e, ll, s, t = self.cnf(X, Z)
        return e, ll, s, t

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), maximize=False, lr=self.hparams.lr, weight_decay=self.hparams.l2, amsgrad=True)
        scheduler = ReduceLROnPlateau(optimizer, factor=.1, patience=3, verbose=False)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_obj"}

    def fit(self, X, Z, max_epochs, batch_size, progress_bar, verbose):
        N = X.shape[0]
        train_size = int(N * 0.9)
        valid_size = N - train_size
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            train, valid = random_split(TensorDataset(X, Z), lengths=[train_size, valid_size])
            train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(valid, batch_size=batch_size)
            trainer = Trainer(
                accelerator="gpu" if torch.cuda.device_count() else "cpu",
                max_epochs=max_epochs,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=progress_bar,
                enable_model_summary=False,
                deterministic=True,
                callbacks=[EarlyStopping(mode='min', monitor='val_obj', min_delta=1e-4, patience=10, verbose=False)],
                gradient_clip_val=1,
                gradient_clip_algorithm="value"
            )
            trainer.fit(model=self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            logs, = trainer.validate(model=self, dataloaders=val_dataloader, verbose=False)
        
        return logs

def ShapiroWilk(e):
    dist = []
    for i in range(e.shape[1]):
        stat, p = shapiro(e[:, i])
        dist.append(stat)
    return -np.array(dist)
    
def HOST(X, eta=1e-4, hidden_sizes=32, lr=1e-1, l1=None, l2=0, max_epochs=30, batch_size=2048, random_state=0, verbose=False, return_dag=False, cutoff=0.001, ci_params=dict(), **kwargs):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
    if random_state is not None:
        torch.random.fork_rng(enabled=True)
        torch.random.manual_seed(random_state)

    N, d = X.shape
    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    X = torch.tensor(X).float()
    l1 = l1 or np.exp(-4e-4 * N + 4)

    perm = []
    layers = []
    remains = np.arange(d)
    while len(remains) > 1:
        cnf = MultipleCNF(d_rem=len(remains), d_cond=len(perm), hidden_sizes=hidden_sizes, lr=lr, l1=l1, l2=l2, verbose=verbose)
        cnf.fit(X=X[:, remains], Z=X[:, perm], max_epochs=max_epochs, batch_size=batch_size, progress_bar=False, verbose=verbose)
        e, ll, s, t = cnf.transform(X=X[:, remains], Z=X[:, perm])
        e, s, t = map(lambda x: x.detach().cpu().numpy(), (e, s, t))
        
        dists = ShapiroWilk(e)
        min_dist = np.nanmin(dists)
        new_nodes_idx, = np.nonzero(np.abs(dists - min_dist) <= eta)
        new_nodes = remains[new_nodes_idx]

        perm.extend(new_nodes[np.argsort(new_nodes)])
        layers.append(new_nodes[np.argsort(new_nodes)].tolist())
        remains = np.delete(remains, new_nodes_idx)

    if len(remains) == 1:
        perm.extend(remains)
        layers.append(list(remains))

    if return_dag:
        return perm, (CIT_perm2dag_parallel(X.numpy(), perm=perm, **ci_params) < cutoff)

    return perm