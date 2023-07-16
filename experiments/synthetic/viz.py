import sys
sys.path.append('')
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib

from src.utils import str_to_matrix
from src.metrics import SHD, AUC, F1, OrderDivergence, PermError, PermErrorPercentage, PermErrorPercentage2

matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
# sns.set()
pd.set_option("display.precision", 2)

def viz_perm_by_N(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.d == 10]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['Perm'] = df['Perm'].apply(lambda x: list(map(int, x.split())))
    df['Order Divergence'] = df.apply(lambda row: OrderDivergence(row['GT'], row['Perm']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)

    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'

    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='N', y='Order Divergence', hue='Method',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 100))
    plt.xscale('log')
    g.set_xlabels(r'\textbf{Sample size}', fontsize=13)
    g.set_ylabels(r'\textbf{Order Divergence ($\downarrow$)}', fontsize=13)
    # g.set_xticklabels(rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', '_by_N.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_perm_by_d(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.N == 500]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['Perm'] = df['Perm'].apply(lambda x: list(map(int, x.split())))
    df['Error (\%)'] = df.apply(lambda row: PermErrorPercentage(row['GT'], row['Perm']), axis=1)
    df['Error'] = df.apply(lambda row: PermError(row['GT'], row['Perm']), axis=1)
    df['Order Divergence'] = df.apply(lambda row: OrderDivergence(row['GT'], row['Perm']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)

    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'

    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='d', y='Order Divergence', hue='Method',
        # style='Whose',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        # order=order,
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 100))
    g.set_xlabels(r'\textbf{Dimensionality}', fontsize=13)
    g.set_ylabels(r'\textbf{Order Divergence ($\downarrow$)}', fontsize=13)
    # g.set_xticklabels(rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', '_by_d.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_dag_by_N(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.d == 10]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['p-values'] = df['p-values'].apply(str_to_matrix)
    df['SHD'] = df.apply(lambda row: SHD(row['GT'], row['p-values'] <= 0.001, row['Method']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)
    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='N', y='SHD', hue='Method',
        # style='Whose',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        # order=order,
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 25))
    g.set_xlabels(r'\textbf{Sample size}', fontsize=13)
    plt.xscale('log')
    g.set_ylabels(r'\textbf{Structural Hamming Distance ($\downarrow$)}', fontsize=11)
    # g.set_xticklabels(order, rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', '_by_N.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_dag_by_d(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.N == 500]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['p-values'] = df['p-values'].apply(str_to_matrix)
    df['SHD'] = df.apply(lambda row: SHD(row['GT'], row['p-values'] <= 0.001, row['Method']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)
    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='d', y='SHD', hue='Method',
        # style='Whose',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        # order=order,
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 25))
    g.set_xlabels(r'\textbf{Dimensionality}', fontsize=13)
    g.set_ylabels(r'\textbf{Structural Hamming Distance ($\downarrow$)}', fontsize=11)
    # g.set_xticklabels(order, rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', 'by_d.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_dag_AUC_by_N(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.d == 10]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['p-values'] = df['p-values'].apply(str_to_matrix)
    df['AUC'] = df.apply(lambda row: AUC(row['GT'], -row['p-values'], row['Method']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)
    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='N', y='AUC', hue='Method',
        # style='Whose',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        # order=order,
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 25))
    g.set_xlabels(r'\textbf{Sample size}', fontsize=13)
    plt.xscale('log')
    g.set_ylabels(r'\textbf{AUC ($\uparrow$)}', fontsize=11)
    # g.set_xticklabels(order, rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', '_by_N_AUC.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_dag_AUC_by_d(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.N == 500]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['p-values'] = df['p-values'].apply(str_to_matrix)
    df['AUC'] = df.apply(lambda row: AUC(row['GT'], -row['p-values'], row['Method']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)
    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='d', y='AUC', hue='Method',
        # style='Whose',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        # order=order,
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 25))
    g.set_xlabels(r'\textbf{Dimensionality}', fontsize=13)
    g.set_ylabels(r'\textbf{AUC ($\uparrow$)}', fontsize=11)
    # g.set_xticklabels(order, rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', 'by_d_AUC.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_dag_F1_by_N(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.d == 10]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['p-values'] = df['p-values'].apply(str_to_matrix)
    df['F1'] = df.apply(lambda row: F1(row['GT'], row['p-values'] <= 0.001, row['Method']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)
    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='N', y='F1', hue='Method',
        # style='Whose',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        # order=order,
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 25))
    g.set_xlabels(r'\textbf{Sample size}', fontsize=13)
    plt.xscale('log')
    g.set_ylabels(r'\textbf{F1 ($\uparrow$)}', fontsize=11)
    # g.set_xticklabels(order, rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', '_by_N_F1.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_dag_F1_by_d(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.N == 500]
    df = df[df.random_state < 5]
    our_name = r'\textbf{HOST (Ours)}'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['GT'] = df['GT'].apply(str_to_matrix)
    df['p-values'] = df['p-values'].apply(str_to_matrix)
    df['F1'] = df.apply(lambda row: F1(row['GT'], row['p-values'] <= 0.001, row['Method']), axis=1)
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)
    print(df.mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'
    g = sns.FacetGrid(df, col='dag_type', sharey=False, sharex=True)
    g.map_dataframe(sns.lineplot, x='d', y='F1', hue='Method',
        # style='Whose',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        # order=order,
        marker='o',
        markersize=6,
        linewidth=2
    )
    # g.set(ylim=(0, 25))
    g.set_xlabels(r'\textbf{Dimensionality}', fontsize=13)
    g.set_ylabels(r'\textbf{F1 ($\uparrow$)}', fontsize=11)
    # g.set_xticklabels(order, rotation=45)
    g.set_titles(r'{col_name}', size=13)
    g.add_legend(label_order=order, fontsize=13)
    
    for ax in g.axes.flat:
        ax.grid(axis='y')
    save_path = path.replace('.csv', 'by_d_F1.pdf')
    g.savefig(save_path)
    print(f'Saved to {save_path}')
    return g

def viz_time(path):
    print(f'{path = }')
    df = pd.read_csv(path)
    df = df[df.random_state < 5]
    our_name = 'HOST (Ours)'
    df['Method'] = df['Method'].map(lambda x: {'OURS': our_name, 'sortnregress': 'VarSort'}.get(x, x))
    df['dag_type'] = df['dag_type'] + '-' + df['k'].astype(str)
    print(df.groupby(['Method', 'N'])['Time'].mean())
    order = df['Method'].unique().tolist()
    order = sorted(order, key=lambda x: {our_name: 'z'}.get(x, x))
    df['Whose'] = our_name
    df.loc[df.Method != our_name, 'Whose'] = 'Existing methods'

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3))
    sns.lineplot(df[df.d == 10], x='N', y='Time', hue='Method',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        hue_order=order,
        marker='o',
        markersize=7,
        linewidth=2,
        legend=False,
        ax=ax0
    )

    sns.lineplot(df[df.N == 500], x='d', y='Time', hue='Method',
        palette={'DiffAN': 'green', 'EqVar': 'blue', 'NPVar': 'orange', 'VarSort': 'yellow', our_name: 'red'},
        hue_order=order,
        marker='o',
        markersize=7,
        linewidth=2,
        legend='brief',
        ax=ax1
    )
    sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))

    ax0.set_xscale('log')
    ax0.set_xlabel(r'\textbf{Sample size}', fontsize=13)
    ax1.set_xlabel(r'\textbf{Dimensionality}', fontsize=13)
    ax0.set_ylabel(r'\textbf{Time (seconds)}', fontsize=13)
    ax1.set_ylabel(r'\textbf{Time (seconds)}', fontsize=13)
    
    ax0.grid(axis='y')
    ax1.grid(axis='y')
    save_path = path.replace('.csv', '_by_time.pdf')
    fig.tight_layout()
    fig.savefig(save_path)
    print(f'Saved to {save_path}')
    return fig