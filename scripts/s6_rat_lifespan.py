
'''
energy lifespan traejctroeis in rat
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

all_metabolic_genes = []
for pathway, genes in energy_dict.items():
    all_metabolic_genes.extend(genes)

all_mito_genes = []
for pathway, genes in mitocarta_dict.items():
    all_mito_genes.extend(genes)

all_genes = list(set(all_metabolic_genes + all_mito_genes))
# 1312 genes


# orthology table was obtained in r using th allen package
orths_allen = pd.read_csv(path_data+
                          'human_orthologs_table_20260324.csv')

orths_allen = orths_allen[['human_Symbol', 
                           'rat_Symbol']]

orths_allen = orths_allen.dropna(subset=['human_Symbol', 'rat_Symbol'])

# safety check to ensure one-to-one orthologs
orths_allen = orths_allen[~orths_allen['human_Symbol'].duplicated(keep=False)]
orths_allen = orths_allen[~orths_allen['rat_Symbol'].duplicated(keep=False)]

energy_orth = orths_allen[orths_allen['human_Symbol'].isin(all_genes)]
# 1242
# unavail_genes = set(all_genes) - set(orths_allen['human_Symbol'])

# make rat energy gene sets
rat_energy_gene_dict = {}
for pathway, genes in energy_dict.items():
    rat_energy_gene_dict[pathway] = \
        orths_allen[orths_allen['human_Symbol'].isin(genes)]['rat_Symbol'].dropna()

rat_mito_gene_dict = {}
for pathway, genes in mitocarta_dict.items():
    rat_mito_gene_dict[pathway] = \
        orths_allen[orths_allen['human_Symbol'].isin(genes)]['rat_Symbol'].dropna()


###################
# expression matrix
# read files
rat_dev_exp = pd.read_csv(path_data+'Rat_rpkm.csv')
rat_dev_exp = rat_dev_exp.drop(columns=['gene_name'])

# convert ensb ids to gene name
biomart = importr('biomaRt')

mart = biomart.useEnsembl("ensembl", 
                          dataset="rnorvegicus_gene_ensembl",
                          version=77)

ensembl_genes = vectors.StrVector(list(rat_dev_exp.Names))

gene_mapping = biomart.getBM(
    attributes=vectors.StrVector(["ensembl_gene_id", "external_gene_name"]),
    filters="ensembl_gene_id",
    values=ensembl_genes,
    mart=mart
)

mapping = pandas2ri.rpy2py(gene_mapping)
mapping.columns = ['ensembl_id', 'gene_symbol']

# save mapping for later
mapping.to_csv(path_result + 'rat_ensembl_to_symbol_mapping.csv', index=False)

# load mapping
mapping = pd.read_csv(path_result + 'rat_ensembl_to_symbol_mapping.csv')

# remove nans and duplicates
mapping = mapping.dropna(subset=['ensembl_id', 'gene_symbol'])
mapping = mapping[~mapping['ensembl_id'].duplicated(keep=False)]
mapping = mapping[~mapping['gene_symbol'].duplicated(keep=False)]

# fix mistmatch in atp5 gene symbols
atp5_genes = [gene for gene in mapping['gene_symbol'] if 'atp5' in gene.lower()]

atp5_rename = pd.read_excel(path_data+'rat_atp5_genes.xlsx')
rename_dict = dict(zip(atp5_rename['alias'], atp5_rename['symbol']))

mapping['gene_symbol'] = mapping['gene_symbol'].replace(rename_dict)

# merge mapping onto expression dataframe
rat_dev_exp = rat_dev_exp.rename(columns={'Names': 'ensembl_id'})
rat_dev_exp1 = rat_dev_exp.merge(mapping, on='ensembl_id', how='inner')
rat_dev_exp1 = rat_dev_exp1.drop(columns=['ensembl_id'])
# 24962 genes

rat_dev_exp = rat_dev_exp1.set_index('gene_symbol').T
# 118 samples x 24,962 genes

# filter to genes with rpkm>=1 in 50 percent of samples
rat_dev_exp1 = rat_dev_exp.loc[:, (rat_dev_exp >= 1).mean() >= (0.5)]
# 12751 genes

# uq normalzie and then log2
percentile_75 = np.percentile(rat_dev_exp1, 75, axis=1)
mean_p75 = np.mean(percentile_75)
rat_dev_uqnorm1 = rat_dev_exp1.div(percentile_75, axis=0) * mean_p75

# log2 transform
rat_dev_uqnorm1 = np.log2(rat_dev_uqnorm1 + 1)

# add relevant columns for age, age-days and region
rat_dev_uqnorm1['age'] = rat_dev_uqnorm1.index.str.split('.').str[1]

# convert age to days
def age_to_days(age_str):
    if age_str.startswith('e'):
        return int(age_str[1:])
    elif age_str.startswith('P'):
        return int(age_str[1:]) + 21

rat_dev_uqnorm1['age_days'] = rat_dev_uqnorm1['age'].apply(age_to_days)
rat_dev_uqnorm1['log_age_days'] = np.log10(rat_dev_uqnorm1['age_days'])
rat_dev_uqnorm1['region'] = rat_dev_uqnorm1.index.str.split('.').str[0]

# make pathway expression matrices
rat_dev_energy_exp = {}
rat_dev_energy_mean = {}
for pathway, genes in rat_energy_gene_dict.items():
    rat_dev_energy_exp[pathway] = rat_dev_uqnorm1.loc[:, rat_dev_uqnorm1.columns.isin(genes)]
    rat_dev_energy_mean[pathway] = np.mean(rat_dev_energy_exp[pathway], axis=1)

rat_dev_energy_mean_df = pd.DataFrame(rat_dev_energy_mean)

# plot
df = rat_dev_energy_mean_df.copy()
df = pd.concat([rat_dev_uqnorm1[['age', 'age_days', 'log_age_days', 'region']], df], axis=1)

# drop ages where theres no forebrain/hindbrain samples
df = df[~df['age'].isin(['e11', 'e12', 'e13'])]

# boxplots
plot_traj_regions_boxes(
    df,
    columns=rat_dev_energy_mean_df.columns,
    regions=['Brain', 'Cerebellum'],
    colors=['mediumvioletred', 'darkseagreen'],
    box=True,
    ncols=5, figsize=(9, 7),
    outpath=path_fig, filename='rat_energy_dev_regions_cardoso.svg'
)
plt.show()


# smooth plots
df_long = pd.melt(df,
                  id_vars=['log_age_days', 'region'],
                  value_vars=rat_dev_energy_mean_df.columns,
                  var_name='pathway',
                  value_name='expression')

df_long['pathway'] = pd.Categorical(df_long['pathway'],
                                    categories=rat_dev_energy_mean_df.columns,
                                    ordered=True)
df_r = pandas2ri.py2rpy(df_long)

x_labels = ['e14', 'e15', 'e16', 'e17', 'e18', 'e19',
    'e20', 'P0', 'P3', 'P7', 'P14', 'P42', 'P112']

plot_energy_loess_cardoso(df_r,
                      x='log_age_days',
                      regions=['Brain', 'Cerebellum'],
                        colors=['mediumvioletred', 'darkseagreen'],
                      x_labels = x_labels,
                      width=11, height=9,
                      vlines=[21],
                      path_fig=path_fig,
                      filename='rat_energy_dev_loess_regions.svg')



##########################
# for mitocarata gene sets

# make pathway expression matrices
rat_dev_mito_exp = {}
rat_dev_mito_mean = {}
for pathway, genes in rat_mito_gene_dict.items():
    rat_dev_mito_exp[pathway] = rat_dev_uqnorm1.loc[:, rat_dev_uqnorm1.columns.isin(genes)]
    rat_dev_mito_mean[pathway] = np.mean(rat_dev_mito_exp[pathway], axis=1)

rat_dev_mito_mean_df = pd.DataFrame(rat_dev_mito_mean)

empty_cols = rat_dev_mito_mean_df.columns[rat_dev_mito_mean_df.isna().all()].tolist()
rat_dev_mito_mean_df.drop(empty_cols, axis=1, inplace=True)

# important pathways for plotting
imp_mito = pd.read_csv(path_data+'imp_mito_plotting.csv', header=None).iloc[:,0].tolist()
rat_dev_mito_mean_df= rat_dev_mito_mean_df[imp_mito]

# plot
df = rat_dev_mito_mean_df.copy()
df = pd.concat([rat_dev_uqnorm1[['age', 'age_days', 'log_age_days', 'region']], df], axis=1)
df = df[~df['age'].isin(['e11', 'e12', 'e13'])]

# boxplots
plot_traj_regions_boxes(
    df,
    columns=rat_dev_mito_mean_df.columns,
    regions=['Brain', 'Cerebellum'],
    colors=['mediumvioletred', 'darkseagreen'],
    box=True,
    ncols=5, figsize=(10.5,9),
    outpath=path_fig, filename='rat_mito_dev_regions_cardoso_final.svg'
)
# plt.show()


# smooth plots
df_long = pd.melt(df,
                  id_vars=['log_age_days', 'region'],
                  value_vars=rat_dev_mito_mean_df.columns,
                  var_name='pathway',
                  value_name='expression')

df_long['pathway'] = pd.Categorical(df_long['pathway'],
                                    categories=rat_dev_mito_mean_df.columns,
                                    ordered=True)
df_r = pandas2ri.py2rpy(df_long)

x_labels = ['e14', 'e15', 'e16', 'e17', 'e18', 'e19',
    'e20', 'P0', 'P3', 'P7', 'P14', 'P42', 'P112']

# both in one
plot_energy_loess_cardoso(df_r,
                      x='log_age_days',
                      regions=['Brain', 'Cerebellum'],
                        colors=['mediumvioletred', 'darkseagreen'],
                      x_labels = x_labels,
                      width=11, height=9,
                      vlines=[21],
                      path_fig=path_fig,
                      filename='rat_mito_dev_loess_regions.svg')


#################
# pc1 replication
rat_dev_energy_pc1 = {}
for pathway, genes in rat_energy_gene_dict.items():
    pca = PCA(n_components=1)
    rat_dev_energy_pc1[pathway] = np.squeeze(pca.fit_transform(zscore(rat_dev_energy_exp[pathway])))
    # check that pc1 is correlated with mean expression of the pathway and flip sign if negatively correlated
    if pearsonr(rat_dev_energy_pc1[pathway], rat_dev_energy_mean[pathway])[0] < 0:
        rat_dev_energy_pc1[pathway] = -rat_dev_energy_pc1[pathway]

rat_dev_energy_pc1_df = pd.DataFrame(rat_dev_energy_pc1)
rat_dev_energy_pc1_df.index = rat_dev_uqnorm1.index

# plot
df = rat_dev_energy_pc1_df.copy()
df = pd.concat([rat_dev_uqnorm1[['age', 'age_days', 'log_age_days', 'region']], df], axis=1)
df = df[~df['age'].isin(['e11', 'e12', 'e13'])]

# boxplot
plot_traj_regions_boxes(
    df,
    columns=rat_dev_energy_pc1_df.columns,
    regions=['Brain', 'Cerebellum'],
    colors=['mediumvioletred', 'darkseagreen'],
    box=True,
    ncols=5, figsize=(10.5, 9),
    outpath=path_fig, filename='rat_energy_dev_regions_cardoso_pc1.svg'
)
plt.show()

# excel with name of genes in each pathway for rat
with pd.ExcelWriter(path_result + 'rat_energy_mito_genes.xlsx') as writer:
    for key, exp_dict in [('energy', rat_dev_energy_exp), ('mitocarta', rat_dev_mito_exp)]:
        df = pd.DataFrame([(pathway, ', '.join(sorted(df.columns))) 
                           for pathway, df in exp_dict.items()],
                          columns=['pathway', 'genes'])
        df.to_excel(writer, sheet_name=key, index=False)