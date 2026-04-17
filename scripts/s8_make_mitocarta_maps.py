
'''
plotting and clustering of mitocarta3.0 maps

author: moohebat
'''

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import colormaps as cmaps
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from statsmodels.stats.multitest import multipletests
from scripts.utils import (pair_corr_spin, 
                           plot_schaefer_fsaverage,
                           load_expression, 
                           filter_expression_ds)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'


'''
setup
'''
# retrieve expression data for schaefer400 parcellation
expression_schaefer400 = load_expression(scale=400)

# save expression dict for filtering
with open(path_data + 'expression_dict_schaefer400.pickle', 'wb') as f:
    pickle.dump(expression_schaefer400, f)

# filter genes with differential stability >0.1
expression_ds01, ds = filter_expression_ds(expression_schaefer400, ds=0.1)
# dataframe of 400 x 8687

# save
with open(path_data + 'expression_ds01.pickle', 'wb') as f:
    pickle.dump(expression_ds01, f)



'''
analysis
'''
with open(path_data + 'expression_ds01.pickle', 'rb') as f:
    expression_ds01 = pickle.load(f)

with open(path_result+'mitocarta_dict.pickle', 'rb') as f:
    mitocarta_dict = pickle.load(f)

# loading schaefer400 spins
spins1k = np.load(path_data+'spins1k.npy')

# make mitocarta expression maps
mito_exp = {}
mito_mean = {}
mito_pc1 = {}
pca = PCA(n_components=1)

for key, value in mitocarta_dict.items():
    mito_exp[key] = expression_ds01[expression_ds01.columns.intersection(value)]
    
    if mito_exp[key].shape[1] == 0:
        print(f"no genes found for '{key}', skippidy")
        continue
    
    mito_mean[key] = np.mean(mito_exp[key], axis=1)

    if mito_exp[key].shape[1] >= 2:
        mito_pc1[key] = np.squeeze(pca.fit_transform(mito_exp[key]))


# save
with open(path_result + 'mito_exp_matrix_400.pickle', 'wb') as f:
    pickle.dump(mito_exp, f)
with open(path_result + 'mito_mean_exp_400.pickle', 'wb') as f:
    pickle.dump(mito_mean, f)
with open(path_result + 'mito_pc1_exp_400.pickle', 'wb') as f:
    pickle.dump(mito_pc1, f)


'''
analysis
'''
# laod genesets
with open(path_result+'mitocarta_dict.pickle', 'rb') as f:
    mitocarta_dict = pickle.load(f)

with open(path_result + 'mito_exp_matrix_400.pickle', 'rb') as f:
    mito_exp = pickle.load(f)
with open(path_result + 'mito_mean_exp_400.pickle', 'rb') as f:
    mito_mean = pickle.load(f)
with open(path_result + 'mito_pc1_exp_400.pickle', 'rb') as f:
    mito_pc1 = pickle.load(f)

# write mitocarta mean expression maps to csv
mito_mean_df = pd.DataFrame(mito_mean)
mito_pc1_df = pd.DataFrame(mito_pc1)

mito_mean_df.to_csv(path_result+'mitocarta_mean_expression_400.csv')
mito_pc1_df.to_csv(path_result+'mitocarta_pc1_expression_400.csv')


# plot mean gene expression
for key, value in mito_mean.items():
    plot_schaefer_fsaverage(zscore(value), cmap=cmaps.matter_r)
    plt.title(key)
    plt.savefig(path_fig+key+'_mean.svg')
    # plt.show()

# plotting pc1 gene expression
for key, value in mito_pc1.items():
    # flip to match mean
    if np.corrcoef(mito_mean[key], value)[0, 1] < 0:
        value = -value
    plot_schaefer_fsaverage(value, cmap=cmaps.BlueWhiteOrangeRed)
    plt.title(key)
    plt.savefig(path_fig+key+'_pc1.svg')
    # plt.show()


# clustering
mito_mean_df = pd.read_csv(path_result+'mitocarta_mean_expression_400.csv').drop(columns='label')
imp_mito = pd.read_csv(path_data+'imp_mito_clustering.csv').iloc[:,0].tolist()

df = mito_mean_df[[x for x in mito_mean_df.columns if x in imp_mito]]

# see if they show clustering
pca = PCA(n_components=2)
pca_exp = pca.fit_transform(df.T)
pc_df = pd.DataFrame(data=pca_exp, columns=['PC1', 'PC2'])
pc_df['pathway'] = df.T.index

plt.figure(figsize=(4,3))
sns.scatterplot(data=pc_df, x='PC1', y='PC2', 
                hue='pathway', s=52, linewidth=0, palette='tab20',
                legend=False)
for line in range(0, pc_df.shape[0]):
    plt.text(pc_df.PC1[line]+0.02, pc_df.PC2[line]+0.02,
            pc_df.pathway[line], horizontalalignment='left',
            fontsize=5, color='black')
plt.title('PCA on energy pathway maps')
sns.despine()
plt.tight_layout()
plt.savefig(path_fig+'pathways_pca_clustering_final.svg')
plt.show()


# hierarchical clustering
correlations = df.corr('spearman')

# run correlation with spin test
mito_corrs, mito_pspins = pair_corr_spin(df, df, spins1k)

mito_corrs.to_csv(path_result+'mitocarta_corrs.csv')
mito_pspins.to_csv(path_result+'mitocarta_pspins.csv')

# fdr correction for multiple testing
model_pval = multipletests(mito_pspins.values.flatten(), method='fdr_bh')[1]
model_pval = pd.DataFrame(model_pval.reshape(28,28))
model_pval.columns = mito_pspins.columns
model_pval.index  = mito_pspins.index

model_pval.to_csv(path_result+'mitocarta_fdr_pval.csv')

# plot
distance = 1 - correlations
np.fill_diagonal(distance.values, 0)  # ensure exact zeros on diagonal
z = linkage(squareform(distance.values), method='average')

g = sns.clustermap(correlations,
                    row_linkage=z,
                    col_linkage=z,
                    cmap=cmaps.BlueWhiteOrangeRed,
                    vmin=-1, vmax=1,
                    square=True,
                    cbar=True,
                    linewidths=0.3,
                    figsize=(10, 10))

# get reordered labels
row_order = g.dendrogram_row.reordered_ind
col_order = g.dendrogram_col.reordered_ind
row_labels = correlations.index[row_order]
col_labels = correlations.columns[col_order]

# reindex pval matrix to match clustermap order
pval_reordered = model_pval.loc[row_labels, col_labels]

# bold edge for significance
ax = g.ax_heatmap
for i in range(pval_reordered.shape[0]):
    for j in range(pval_reordered.shape[1]):
        if i == j:
            continue
        if pval_reordered.iloc[i, j] < 0.05:
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1,
                fill=False,
                edgecolor='black',
                lw=1.5,
                clip_on=False
            ))
plt.tight_layout()
plt.savefig(path_fig+f'corr_clustermap_significance.svg')
plt.show()



# class enrichment heatmap
# loading class labels
yeo_schaefer400 = np.load(path_data+'yeo_schaefer400.npy')
ve_schaefer400 = np.load(path_data+'ve_schaefer400.npy', allow_pickle=True)
mesulam_schaefer400 = np.load(path_data+'mesulam_schaefer400.npy', allow_pickle=True)

mapping = {
    'Cont': 'FP',
    'Default': 'DM',
    'DorsAttn': 'DA',
    'Limbic': 'Lim',
    'SalVentAttn': 'SA',
    'SomMot': 'SM',
    'Vis': 'Vis'
}
yeo_schaefer400 = np.array([mapping[val] for val in yeo_schaefer400])

# ve
mapping = {
    'association':'Ac',
    'association2':'Ac2',
    'insular': 'Ins',
    'limbic': 'Lim',
    'primary motor': 'PM',
    'primary sensory': 'PS',
    'primary/secondary sensory': 'PSS'
}
ve_schaefer400 = np.array([mapping[val] for val in ve_schaefer400])

# mesulam
mapping = {
    'hetermodal': 'HM', 
    'idiotypic': 'ID', 
    'paralimbic': 'PLB', 
    'unimodal': 'UM'
}
mesulam_schaefer400 = np.array([mapping[val] for val in mesulam_schaefer400])

classes = {'yeo':{
    'order': ['SM','Vis', 'DA', 'SA', 'FP','DM','Lim'], 
    'map': yeo_schaefer400},
    've': {
        'order': ['PM','PS','PSS', 'Ac','Ac2','Ins','Lim'],
        'map': ve_schaefer400},
    'mesulam': {
        'order': ['ID', 'UM', 'HM', 'PLB'],
        'map': mesulam_schaefer400}}

def class_enrichment_heatmap(data, class_labels, spins, order, outpath, filename):
    nets = order
    mapz = pd.DataFrame(zscore(data), columns=data.columns)
    
    emp_matrix = pd.DataFrame(index=data.columns, columns=nets, dtype=float)
    pval_matrix = pd.DataFrame(index=data.columns, columns=nets, dtype=float)
    
    for net in nets:
        mask = (class_labels == net)
        for pathway in mapz.columns:
            pathway_vals = mapz[pathway].values
            emp = np.mean(pathway_vals[mask])
            nulls = np.array([np.mean(pathway_vals[spins[:, i]][mask]) 
                              for i in range(spins.shape[1])])
            pval = (1 + np.sum(np.abs(nulls - np.mean(nulls)) >= 
                               abs(emp - np.mean(nulls)))) / (spins.shape[1] + 1)
            emp_matrix.loc[pathway, net] = emp
            pval_matrix.loc[pathway, net] = pval

    pvals_flat = pval_matrix.values.flatten()
    _, pvals_adj, _, _ = multipletests(pvals_flat, method='fdr_bh')
    pval_adj = pd.DataFrame(pvals_adj.reshape(pval_matrix.shape),
                                         index=pval_matrix.index,
                                         columns=pval_matrix.columns)
    sig_matrix = pval_adj < 0.05
    annot = sig_matrix.applymap(lambda x: '*' if x else '')
    
    fig, ax = plt.subplots(figsize=(len(nets) * 1, len(mapz.columns) * 0.5))
    sns.heatmap(emp_matrix.astype(float),
                annot=annot, fmt='',
                cmap=cmaps.matter_r, center=0,
                linewidths=0.3,
                ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(outpath + filename + '.svg')
    plt.show()
    
    return emp_matrix, pval_adj

# run
mito_emp, mito_pval = class_enrichment_heatmap(df,
                                                ve_schaefer400,
                                                spins1k,
                                                classes['ve']['order'],
                                                path_fig,
                                                've_mito_heatmap')

mito_emp, mito_pval = class_enrichment_heatmap(df,
                                                yeo_schaefer400,
                                                spins1k,
                                                classes['yeo']['order'],
                                                path_fig,
                                                'yeo_mito_heatmap')