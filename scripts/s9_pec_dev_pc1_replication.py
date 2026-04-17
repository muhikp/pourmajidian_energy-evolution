
'''
Script to replicate energy metabolism and 
MitoCarta3.0 gene expression trajectories
in human and macaque using PsychENCODE evolution data

author: moohebat
'''

# importy import
import numpy as np
import pandas as pd
import pickle
import colormaps as cmaps
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scripts.utils import plot_energy_trajectories, plot_energy_loess

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib import grdevices
from rpy2.robjects import pandas2ri
pandas2ri.activate()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'


# load files
# pec dev data
pec_dev_exp = pd.read_csv(path_result + 'pec_dev_exp_uqnorm.csv', index_col=0)
pec_dev_sample_info = pd.read_csv(path_result + 'pec_dev_sample_info_qc.csv', index_col=0)

# focusing on cortical samples
ncx_idx = pec_dev_sample_info[pec_dev_sample_info['NCXRegion'] == 'NCX'].index
pec_dev_sample_info = pec_dev_sample_info.loc[ncx_idx]
pec_dev_exp = pec_dev_exp.loc[ncx_idx]

# load energy gene sets
with open(path_result+'energy_genelist_dict_consolidated.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

with open(path_result+'mitocarta_dict.pickle', 'rb') as f:
    mitocarta_dict = pickle.load(f)


################################
# PEC uses old atp5 gene symbols

# find atp5 genes in pec dev data
atp5_genes = [col for col in pec_dev_exp.columns if col.startswith('ATP5')]

# convert to new atp5 gene symbols
# mapping was downloaded from SynGO
atp5_mapping = pd.read_excel(path_data + 'pec_atp5_id_conversion.xlsx')
atp5_mapping = atp5_mapping[['query', 'symbol']]

atp5_dict = dict(zip(atp5_mapping['query'], atp5_mapping['symbol']))

# replace old with new in pec dev data
pec_dev_exp = pec_dev_exp.rename(columns=atp5_dict)

# check conversion worked:
[col for col in pec_dev_exp.columns if col.startswith('ATP5')]

extended_maps = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate',
                 'complex1', 'complex2', 'complex3', 'complex4','atpsynth', 
                 'kb_util', 'fa_metabolism', 'glycogen_metabolism', 
                 'pdc', 'mas', 'gps', 'creatine', 'ros_detox', 'ros_gen',
                 'no_signalling', 'atpase', 'gln_glu_cycle', 'bcaa_cat']

# keep only non-redundant pathways
energy_dict = {k: list(energy_dict[k]) for k in extended_maps if k in energy_dict}

# to keep the order of stuff
mac_age_order = ['E60', 'E80', 'E81', 'E82', 'E110', 'E111', 
                 'P0', 'P2', '7M', '1Y', '2Y', '3.5Y', '4Y', '5Y',
                 '7Y', '11Y']

human_age_order = ['8 PCW', '9 PCW', '12 PCW', '13 PCW', '16 PCW', 
                   '17 PCW', '19 PCW', '21 PCW', '22 PCW', '37 PCW',
                   '4 M', '10 M', '1 Y', '3 Y', '4 Y', '8 Y', 
                   '11 Y', '13 Y', '15 Y', '19 Y', 
                   '21 Y', '23 Y', '30 Y', '36 Y', '37 Y', '40 Y']

names1 = ['prenate','infant','child','adolescent','adult']

names2 = ['early_fetal','mid_fetal','late_fetal','infant', 
         'early_child','late_child','adolescent','adult']

pec_dev_sample_info['Age'] = pd.Categorical(
    pec_dev_sample_info['Age'],
    categories=mac_age_order + human_age_order,
    ordered=True
)

pec_dev_sample_info['age_group'] = pd.Categorical(
    pec_dev_sample_info['age_group'],
    categories=names1,
    ordered=True
)

pec_dev_sample_info['age_group2'] = pd.Categorical(
    pec_dev_sample_info['age_group2'],
    categories=names2, 
    ordered=True
)

pec_dev_exp_full = pd.concat([pec_dev_sample_info, pec_dev_exp],
                             axis=1)


# 1. my energy pathways
pec_energy_exp = {}
pec_energy_mean = {}
for key, value in energy_dict.items():
        pec_energy_exp[key] = pec_dev_exp.loc[:, pec_dev_exp.columns.isin(value)]
        pec_energy_mean[key] = np.mean(pec_energy_exp[key], axis=1)

# fit pca separately per species
pc_dict_human = {}
pc_dict_mac = {}

for pathway, value in pec_energy_exp.items():
    n_components = min(3, value.shape[1])
    
    human_idx = pec_dev_sample_info['Species'] == 'Human'
    mac_idx = pec_dev_sample_info['Species'] == 'Macaque'
    
    pca_human = PCA(n_components=n_components)
    pc_dict_human[pathway] = pd.DataFrame(
        pca_human.fit_transform(StandardScaler().fit_transform(value[human_idx])),
        index=pec_dev_sample_info[human_idx].index,
        columns=[f'PC{i+1}' for i in range(n_components)])
    
    pca_mac = PCA(n_components=n_components)
    pc_dict_mac[pathway] = pd.DataFrame(
        pca_mac.fit_transform(StandardScaler().fit_transform(value[mac_idx])),
        index=pec_dev_sample_info[mac_idx].index,
        columns=[f'PC{i+1}' for i in range(n_components)])
    
    # flip pc1 sign if negatively correlated with mean expression
    human_mean = value[human_idx].mean(axis=1)
    if np.corrcoef(pc_dict_human[pathway]['PC1'], human_mean)[0, 1] < 0:
        pc_dict_human[pathway]['PC1'] = -pc_dict_human[pathway]['PC1']
    
    mac_mean = value[mac_idx].mean(axis=1)
    if np.corrcoef(pc_dict_mac[pathway]['PC1'], mac_mean)[0, 1] < 0:
        pc_dict_mac[pathway]['PC1'] = -pc_dict_mac[pathway]['PC1']


# build pc1 df for both species
pc1_human = pd.DataFrame({pathway: pc_dict_human[pathway]['PC1'] 
                           for pathway in pc_dict_human})
pc1_mac = pd.DataFrame({pathway: pc_dict_mac[pathway]['PC1'] 
                         for pathway in pc_dict_mac})
pc1_df = pd.concat([pc1_mac, pc1_human])

# plot pc1 trajectories
for x_axis in ['log_age_days', 'log_age_days_predicted']:
    df_long = pd.melt(
        pd.concat([pec_dev_sample_info[[x_axis, 'Species']], pc1_df], axis=1),
        id_vars=[x_axis, 'Species'],
        value_vars=list(pc_dict_human.keys()),
        var_name='pathway',
        value_name='expression')
    df_long['pathway'] = pd.Categorical(df_long['pathway'],
                                        categories=list(pc_dict_human.keys()),
                                        ordered=True)
    df_r = pandas2ri.py2rpy(df_long)
    plot_energy_loess(df_r,
                      x=x_axis,
                      color='Species',
                      colors=['orangered', 'cornflowerblue'],
                      width=11, height=9,
                      path_fig=path_fig,
                      filename=f'energy_dev_loess_{x_axis}_pc1_ctx_zscore.svg')


########################
# for mitocarta pathways
pec_mito_exp = {}
pec_mito_mean = {}
for key, value in mitocarta_dict.items():
        pec_mito_exp[key] = pec_dev_exp.loc[:, pec_dev_exp.columns.isin(value)]
        pec_mito_mean[key] = np.mean(pec_mito_exp[key], axis=1)

# get rid of na pathways
pec_mito_exp = {k: v for k, v in pec_mito_exp.items() if v.shape[1] > 0}

# keep important pathways for plotting
imp_mito = pd.read_csv(path_data+'imp_mito_plotting.csv', header=None).iloc[:,0].tolist()
pec_mito_exp = {k: v for k, v in pec_mito_exp.items() if k in imp_mito}

# fit pca separately per species
pc_mito_dict_human = {}
pc_mito_dict_mac = {}

for pathway, value in pec_mito_exp.items():
    n_components = min(3, value.shape[1])
    
    human_idx = pec_dev_sample_info['Species'] == 'Human'
    mac_idx = pec_dev_sample_info['Species'] == 'Macaque'
    
    pca_human = PCA(n_components=n_components)
    pc_mito_dict_human[pathway] = pd.DataFrame(
        pca_human.fit_transform(StandardScaler().fit_transform(value[human_idx])),
        index=pec_dev_sample_info[human_idx].index,
        columns=[f'PC{i+1}' for i in range(n_components)])
    
    pca_mac = PCA(n_components=n_components)
    pc_mito_dict_mac[pathway] = pd.DataFrame(
        pca_mac.fit_transform(StandardScaler().fit_transform(value[mac_idx])),
        index=pec_dev_sample_info[mac_idx].index,
        columns=[f'PC{i+1}' for i in range(n_components)])
    
    # flip pc1 sign if negatively correlated with mean expression
    human_mean = value[human_idx].mean(axis=1)
    if np.corrcoef(pc_mito_dict_human[pathway]['PC1'], human_mean)[0, 1] < 0:
        pc_mito_dict_human[pathway]['PC1'] = -pc_mito_dict_human[pathway]['PC1']
    
    mac_mean = value[mac_idx].mean(axis=1)
    if np.corrcoef(pc_mito_dict_mac[pathway]['PC1'], mac_mean)[0, 1] < 0:
        pc_mito_dict_mac[pathway]['PC1'] = -pc_mito_dict_mac[pathway]['PC1']

# build pc1 df for both species
pc1_human = pd.DataFrame({pathway: pc_mito_dict_human[pathway]['PC1'] 
                           for pathway in pc_mito_dict_human})
pc1_mac = pd.DataFrame({pathway: pc_mito_dict_mac[pathway]['PC1'] 
                         for pathway in pc_mito_dict_mac})
pc1_df = pd.concat([pc1_mac, pc1_human])

# plot
for x_axis in ['log_age_days', 'log_age_days_predicted']:
    df_long = pd.melt(
        pd.concat([pec_dev_sample_info[[x_axis, 'Species']], pc1_df], axis=1),
        id_vars=[x_axis, 'Species'],
        value_vars=list(pc_mito_dict_human.keys()),
        var_name='pathway',
        value_name='expression')
    df_long['pathway'] = pd.Categorical(df_long['pathway'],
                                        categories=list(pc_mito_dict_human.keys()),
                                        ordered=True)
    df_r = pandas2ri.py2rpy(df_long)
    plot_energy_loess(df_r,
                      x=x_axis,
                      color='Species',
                      colors=['#F15B14', '#698EC9'],
                      width=11, height=7.5,
                      path_fig=path_fig,
                      filename=f'mito_dev_loess_{x_axis}_pc1_ctx_select_zscore.svg')