import sys
sys.path.append('')
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib

from src.utils import str_to_matrix
from src.metrics import SHD, OrderDivergence

matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
# sns.set()
pd.set_option("display.precision", 2)

def viz_perm(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    our_name = 'HOST (Ours)'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['Perm'] = df['Perm'].apply(lambda x: list(map(int, x.split())))
    df['Order Divergence'] = df.apply(lambda row: OrderDivergence(row['GT'], row['Perm']), axis=1)
    print(df.groupby(['dataset', 'Method']).mean())
    print(df.groupby(['dataset', 'Method']).std())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dataset', sharey=True, sharex=True)
    g.map_dataframe(sns.boxplot, x='Method', y='Order Divergence', hue='Whose', dodge=False,
        palette={'Existing methods': 'royalblue', our_name: 'red'},
        order=order,
    )
    g.set_xlabels(r'\textbf{Method}', fontsize=13)
    g.set_ylabels(fontsize=13)
    g.set_xticklabels(order, rotation=45)
    g.set_titles(r'Dataset: {col_name}', size=13)
    g.add_legend(fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', '.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_dag(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    our_name = 'HOST (Ours)'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['p-values'] = df['p-values'].apply(str_to_matrix)
    df['SHD'] = df.apply(lambda row: SHD(row['GT'], row['p-values'] <= 0.001, row['Method']), axis=1)
    print(df.groupby(['dataset', 'Method'])['SHD'].mean())
    print(df.groupby(['dataset', 'Method']).std())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dataset', sharey=True, sharex=True)
    g.map_dataframe(sns.boxplot, x='Method', y='SHD', hue='Whose', dodge=False,
        palette={'Existing methods': 'royalblue', our_name: 'red'},
        order=order,
    )
    g.set_xlabels(r'\textbf{Method}', fontsize=13)
    g.set_ylabels(fontsize=13)
    g.set_xticklabels(order, rotation=45)
    g.set_titles(r'Dataset: {col_name}', size=13)
    g.add_legend(fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', '.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g