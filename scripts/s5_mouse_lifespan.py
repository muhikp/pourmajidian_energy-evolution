
'''
energy lifespan traejctroeis in mouse
using the cardoso-moreira 2019 dataset

author: moohebat
'''

import numpy as np
import pandas as pd
import pickle
from scipy.stats import zscore, pearsonr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scripts.utils import (plot_energy_loess_cardoso, 
                           plot_traj_regions_boxes)

import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import vectors
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib import grdevices

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

'''
make ortholog gene list
'''
# load energy gene sets
with open(path_result+'energy_genelist_dict_consolidated.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# save mitocartsa dict
with open(path_result+'mitocarta_dict.pickle', 'rb') as f:
    mitocarta_dict = pickle.load(f)

# keep only non-redundant pathways
extended_maps = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate',
                 'complex1', 'complex2', 'complex3', 'complex4','atpsynth', 
                 'kb_util', 'fa_metabolism', 'glycogen_metabolism', 
                 'pdc', 'mas', 'gps', 'creatine', 'ros_detox', 'ros_gen',
                 'no_signalling', 'atpase', 'gln_glu_cycle', 'bcaa_cat']

energy_dict = {k: list(energy_dict[k]) for k in extended_maps if k in energy_dict}

all_energy_genes = []
for pathway, genes in energy_dict.items():
    all_energy_genes.extend(genes)

all_mito_genes = []
for pathway, genes in mitocarta_dict.items():
    all_mito_genes.extend(genes)

all_genes = list(set(all_energy_genes + all_mito_genes))
# 1312 genes


# orthology table was obtained in r using th allen package
orths_allen = pd.read_csv(path_data+
                          'human_orthologs_table_20260324.csv')

orths_allen = orths_allen[['human_Symbol', 'mouse_Symbol']]


orths_allen = orths_allen.dropna(subset=['human_Symbol', 'mouse_Symbol'])

# safety check to ensure one-to-one orthologs
orths_allen = orths_allen[~orths_allen['human_Symbol'].duplicated(keep=False)]
orths_allen = orths_allen[~orths_allen['mouse_Symbol'].duplicated(keep=False)]

energy_orth = orths_allen[orths_allen['human_Symbol'].isin(all_genes)]
# 1236
# unavail_genes = set(all_genes) - set(orths_allen['human_Symbol'])

# make mouse energy gene sets
mouse_energy_gene_dict = {}
for pathway, genes in energy_dict.items():
    # get pathway ortholog gene sets
    mouse_energy_gene_dict[pathway] = \
        orths_allen[orths_allen['human_Symbol'].isin(genes)]['mouse_Symbol'].dropna()

# mouse mitocarta genestes
mouse_mito_gene_dict = {}
for pathway, genes in mitocarta_dict.items():
    # get pathway ortholog gene sets
    mouse_mito_gene_dict[pathway] = \
        orths_allen[orths_allen['human_Symbol'].isin(genes)]['mouse_Symbol'].dropna()

###################
# expression matrix
# read files
mouse_dev_exp = pd.read_csv(path_data+'Mouse_rpkm.csv')

# convert ensb ids to gene name
biomart = importr('biomaRt')

mart = biomart.useEnsembl("ensembl", 
                          dataset="mmusculus_gene_ensembl",
                          version=77)

ensembl_genes = vectors.StrVector(list(mouse_dev_exp.Names))

gene_mapping = biomart.getBM(
    attributes=vectors.StrVector(["ensembl_gene_id", "external_gene_name"]),
    filters="ensembl_gene_id",
    values=ensembl_genes,
    mart=mart
)

mapping = pandas2ri.rpy2py(gene_mapping)
mapping.columns = ['ensembl_id', 'gene_symbol']

# save mapping so i dont have to run biomart again
mapping.to_csv(path_result + 'mouse_ensembl_to_symbol_mapping.csv', index=False)

# load mapping
mapping = pd.read_csv(path_result + 'mouse_ensembl_to_symbol_mapping.csv')

# remove nans and duplicates
mapping = mapping.dropna(subset=['ensembl_id', 'gene_symbol'])
mapping = mapping[~mapping['ensembl_id'].duplicated(keep=False)]
mapping = mapping[~mapping['gene_symbol'].duplicated(keep=False)]

# merge mapping onto expression dataframe
mouse_dev_exp = mouse_dev_exp.rename(columns={'Names': 'ensembl_id'})
mouse_dev_exp1 = mouse_dev_exp.merge(mapping, on='ensembl_id', how='inner')
mouse_dev_exp1 = mouse_dev_exp1.drop(columns=['ensembl_id'])
# 35935 genes

mouse_dev_exp = mouse_dev_exp1.set_index('gene_symbol').T
# 98 samples x 35935 genes

# filter to genes with rpkm>=1 in 50 percent of samples
mouse_dev_exp1 = mouse_dev_exp.loc[:, (mouse_dev_exp >= 1).mean() >= (0.5)]
# 12983 genes

# uq normalzie and then log2
percentile_75 = np.percentile(mouse_dev_exp1, 75, axis=1)
mean_p75 = np.mean(percentile_75)
mouse_dev_uqnorm1 = mouse_dev_exp1.div(percentile_75, axis=0) * mean_p75

# log2 transform
mouse_dev_uqnorm1 = np.log2(mouse_dev_uqnorm1 + 1)

# add relevant columns for age, age-days and region
mouse_dev_uqnorm1['age'] = mouse_dev_uqnorm1.index.str.split('.').str[1]

# convert age to days
def convert_age_to_days(age):
    if age.startswith('e'):
        return int(age[1:])
    elif age.startswith('P'):
        return int(age[1:])+20 # gestation is approx 20 days

mouse_dev_uqnorm1['age_days'] = mouse_dev_uqnorm1['age'].apply(convert_age_to_days)
mouse_dev_uqnorm1['log_age_days'] = np.log10(mouse_dev_uqnorm1['age_days'])

mouse_dev_uqnorm1['region'] = mouse_dev_uqnorm1.index.str.split('.').str[0]


# make pathway expression matrices
mouse_dev_energy_exp = {}
mouse_dev_energy_mean = {}
for pathway, genes in mouse_energy_gene_dict.items():
    mouse_dev_energy_exp[pathway] = mouse_dev_uqnorm1.loc[:, mouse_dev_uqnorm1.columns.isin(genes)]
    mouse_dev_energy_mean[pathway] = np.mean(mouse_dev_energy_exp[pathway], axis=1)

mouse_dev_energy_mean_df = pd.DataFrame(mouse_dev_energy_mean)

# plot
df = mouse_dev_energy_mean_df.copy()

df = pd.concat([mouse_dev_uqnorm1[['age', 'age_days', 'log_age_days', 'region']], df], axis=1)

# drop samples where age is 'e10', 'e11', 'e12'
# because theres no forebrain/hindbrain samples
df = df[~df['age'].isin(['e10', 'e11', 'e12'])]

# boxplots
plot_traj_regions_boxes(
    df,
    columns=mouse_dev_energy_mean_df.columns,
    regions=['Brain', 'Cerebellum'],
    colors=['mediumvioletred', 'darkseagreen'],
    box=True,
    ncols=5, figsize=(9, 7),
    outpath=path_fig, filename='mouse_energy_dev_regions_cardoso.svg'
)
plt.show()


# smooth plots
df_long = pd.melt(df,
                  id_vars=['log_age_days', 'region'],
                  value_vars=mouse_dev_energy_mean_df.columns,
                  var_name='pathway',
                  value_name='expression')

# keep order of pathways in the plot
df_long['pathway'] = pd.Categorical(df_long['pathway'],
                                    categories=mouse_dev_energy_mean_df.columns,
                                    ordered=True)

df_r = pandas2ri.py2rpy(df_long)

x_labels = ['e13', 'e14', 'e15', 'e16', 'e17', 'e18',
    'P0', 'P3', 'P14', 'P28', 'P63']

# both in one
plot_energy_loess_cardoso(df_r,
                      x='log_age_days',
                      regions=['Brain', 'Cerebellum'],
                        colors=['mediumvioletred', 'darkseagreen'],
                      x_labels = x_labels,
                      width=11, height=9,
                      vlines=[20],
                      path_fig=path_fig,
                      filename='mouse_energy_dev_loess_regions.svg')



##########################
# for mitocarata gene sets

# load mouse mitocarta 3.0 gene list
mito_pathway = pd.read_csv(path_data + 'mitocarta_pathways_mouse.csv')

# get atp5 genes in the expression data
atp_genes = [col for col in mouse_dev_uqnorm1.columns if col.startswith('Atp5')]
# uses old nomenclature, same as mitocarta, so its foine

mitocarta_dict = dict(zip(mito_pathway['MitoPathway'], 
                            mito_pathway['Genes'].str.split(', ')))


# make pathway expression matrices
mouse_dev_mito_exp = {}
mouse_dev_mito_mean = {}
for pathway, genes in mitocarta_dict.items():
    mouse_dev_mito_exp[pathway] = mouse_dev_uqnorm1.loc[:, mouse_dev_uqnorm1.columns.isin(genes)]
    mouse_dev_mito_mean[pathway] = np.mean(mouse_dev_mito_exp[pathway], axis=1)

mouse_dev_mito_mean_df = pd.DataFrame(mouse_dev_mito_mean)

empty_cols = mouse_dev_mito_mean_df.columns[mouse_dev_mito_mean_df.isna().all()].tolist()
mouse_dev_mito_mean_df.drop(empty_cols, axis=1, inplace=True)

# keep important pathways for plotting
imp_mito = pd.read_csv(path_data+'imp_mito_plotting.csv', header=None).iloc[:,0].tolist()
mouse_dev_mito_mean_df= mouse_dev_mito_mean_df[imp_mito]

# plot
df = mouse_dev_mito_mean_df.copy()
df = pd.concat([mouse_dev_uqnorm1[['age', 'age_days', 'log_age_days', 'region']], df], axis=1)

df = df[~df['age'].isin(['e10', 'e11', 'e12'])]

# boxplots for lifespan trajectories
plot_traj_regions_boxes(
    df,
    columns=mouse_dev_mito_mean_df.columns,
    regions=['Brain', 'Cerebellum'],  # or add a whole-brain category
    colors=['mediumvioletred', 'darkseagreen'],
    box=True,
    ncols=5, figsize=(10.5, 9),
    outpath=path_fig, filename='mouse_mito_dev_regions_cardoso_final.svg'
)
plt.show()


# smooth plots
df_long = pd.melt(df,
                  id_vars=['log_age_days', 'region'],
                  value_vars=mouse_dev_mito_mean_df.columns,
                  var_name='pathway',
                  value_name='expression')

# keep order of pathways in the plot
df_long['pathway'] = pd.Categorical(df_long['pathway'],
                                    categories=mouse_dev_mito_mean_df.columns,
                                    ordered=True)

df_r = pandas2ri.py2rpy(df_long)

x_labels = ['e13', 'e14', 'e15', 'e16', 'e17', 'e18',
    'P0', 'P3', 'P14', 'P28', 'P63']

# both in one
plot_energy_loess_cardoso(df_r,
                      x='log_age_days',
                      regions=['Brain', 'Cerebellum'],
                        colors=['mediumvioletred', 'darkseagreen'],
                      x_labels = x_labels,
                      width=11, height=9,
                      vlines=[20],
                      path_fig=path_fig,
                      filename='mouse_mito_dev_loess_regions.svg')

#################
# pc1 replication
# make pathway expression matrices
pca = PCA(n_components=1)
mouse_dev_energy_pc1 = {}
for pathway, genes in mouse_energy_gene_dict.items():
    mouse_dev_energy_pc1[pathway] = np.squeeze(pca.fit_transform(zscore(mouse_dev_energy_exp[pathway])))
    # check that pc1 is correlated with mean expression of the pathway and flip sign if negatively correlated
    if pearsonr(mouse_dev_energy_pc1[pathway], mouse_dev_energy_mean[pathway])[0] < 0:
        mouse_dev_energy_pc1[pathway] = -mouse_dev_energy_pc1[pathway]


mouse_dev_energy_pc1_df = pd.DataFrame(mouse_dev_energy_pc1)
mouse_dev_energy_pc1_df.index = mouse_dev_uqnorm1.index

# plot
df = mouse_dev_energy_pc1_df.copy()
df = pd.concat([mouse_dev_uqnorm1[['age', 'age_days', 'log_age_days', 'region']], df], axis=1)

df = df[~df['age'].isin(['e10', 'e11', 'e12'])]

# boxplots
plot_traj_regions_boxes(
    df,
    columns=mouse_dev_energy_pc1_df.columns,
    regions=['Brain', 'Cerebellum'],  # or add a whole-brain category
    colors=['mediumvioletred', 'darkseagreen'],
    box=True,
    ncols=5, figsize=(9, 7),
    outpath=path_fig, filename='mouse_energy_dev_regions_cardoso_pc1.svg'
)
plt.show()


# excel with name of genes in each pathway for mouse
with pd.ExcelWriter(path_result + 'mouse_energy_mito_genes.xlsx') as writer:
    for key, exp_dict in [('energy', mouse_dev_energy_exp), ('mitocarta', mouse_dev_mito_exp)]:
        df = pd.DataFrame([(pathway, ', '.join(sorted(df.columns))) 
                           for pathway, df in exp_dict.items()],
                          columns=['pathway', 'genes'])
        df.to_excel(writer, sheet_name=key, index=False)