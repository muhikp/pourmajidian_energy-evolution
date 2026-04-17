
'''
Script to analyze energy metabolism and 
MitoCarta3.0 gene expression trajectories
in human and macaque using PsychENCODE evolution data

author: moohebat
'''

# importy import
import numpy as np
import pandas as pd
import pickle
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import plot_energy_trajectories, plot_energy_loess

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib import grdevices
from rpy2.robjects import pandas2ri
from rpy2.robjects import r
pandas2ri.activate()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load pec dev data
pec_dev_exp = pd.read_csv(path_result + 'pec_dev_exp_uqnorm.csv', index_col=0)
pec_dev_sample_info = pd.read_csv(path_result + 'pec_dev_sample_info_qc.csv', index_col=0)

# load energy gene sets
with open(path_result+'energy_genelist_dict_consolidated.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# load mitocarta dict
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

# convert to dict
atp5_dict = dict(zip(atp5_mapping['query'], atp5_mapping['symbol']))

# replace
pec_dev_exp = pec_dev_exp.rename(columns=atp5_dict)

# check conversion worked:
[col for col in pec_dev_exp.columns if col.startswith('ATP5')]

extended_maps = ['glycolysis', 'ppp', 'tca', 'oxphos', 'lactate',
                 'complex1', 'complex2', 'complex3', 'complex4','atpsynth', 
                 'kb_util', 'fa_metabolism', 'glycogen_metabolism', 'bcaa_cat'
                 'pdc', 'mas', 'gps', 'creatine', 'ros_detox', 'ros_gen',
                 'no_signalling', 'atpase', 'gln_glu_cycle']


# keep only non-redundant pathways
energy_dict = {k: list(energy_dict[k]) for k in extended_maps if k in energy_dict}


###################
# lifespan analysis
# to keep the order of age when plotting
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


######################
# main energy pathways

# put expression matrix and sample info together
pec_dev_exp_full = pd.concat([pec_dev_sample_info, pec_dev_exp],
                             axis=1)

# make energy expression matrices
pec_energy_exp = {}
pec_energy_mean = {}
for key, value in energy_dict.items():
        pec_energy_exp[key] = pec_dev_exp.loc[:, pec_dev_exp.columns.isin(value)]
        pec_energy_mean[key] = np.mean(pec_energy_exp[key], axis=1)

pec_energy_mean_df = pd.DataFrame(pec_energy_mean)

# add sample info for plotting
energy_df = pd.concat([pec_dev_sample_info, pec_energy_mean_df], axis=1)

# macaque boxplot
fig_size = {'age_group': (9, 8), 'Age': (10, 8)}
rotations = {'age_group': 35, 'Age': 90}
categories = ['age_group', 'Age']

# only cortical
# 579 samples
for category in categories:
    plot_energy_trajectories(energy_df[energy_df['NCXRegion'] == 'NCX'], 
                            species='Macaque', 
                            columns=pec_energy_mean_df.columns,
                            category=category,
                            rotations=rotations[category],
                        figsize=fig_size[category],
                        outpath=path_fig,
                        filename=f"mac_energy_traj_{category}_ctx.svg")

# smooth plots
for x_axis in ['log_age_days']:

    df_long = pd.melt(energy_df[energy_df['NCXRegion'] == 'NCX'], 
                    id_vars=[x_axis, 'Species'],
                    value_vars=pec_energy_mean_df.columns,
                    var_name='pathway', 
                    value_name='expression')

    df_long['pathway'] = pd.Categorical(df_long['pathway'], 
                                        categories=pec_energy_mean_df.columns, 
                                        ordered=True)
    df_r = pandas2ri.py2rpy(df_long)

    plot_energy_loess(df_r, 
                    x=x_axis, 
                    path_fig=path_fig, 
                    colors=['#F15B14', '#698EC9'],
                    width=11, height=9,
                    filename=f"both_energy_dev_loess_{x_axis}_ctx.svg")


###########
# mitocarta
# make mitocarta expression matrices
pec_mitocarta_exp = {}
pec_mitocarta_mean = {}
for key, value in mitocarta_dict.items():
    pec_mitocarta_exp[key] = pec_dev_exp.loc[:, pec_dev_exp.columns.isin(value)]
    pec_mitocarta_mean[key] = np.mean(pec_mitocarta_exp[key], axis=1)

pec_mito_mean_df = pd.DataFrame(pec_mitocarta_mean)
pec_mito_mean_df = pec_mito_mean_df.dropna(axis=1, how='all')

# keep important pathways for plotting
imp_mito = pd.read_csv(path_data+'imp_mito_plotting.csv', header=None).iloc[:,0].tolist()
pec_mito_mean_df= pec_mito_mean_df[imp_mito]
# 29 pathways

# add sample info for plotting
mito_df = pd.concat([pec_dev_sample_info, pec_mito_mean_df], axis=1)

# smooth plots
# r('install.packages("svglite")')
for x_axis in ['log_age_days']:

    df_long = pd.melt(mito_df[mito_df['NCXRegion'] == 'NCX'], 
                    id_vars=[x_axis, 'Species'],
                    value_vars=pec_mito_mean_df.columns,
                    var_name='pathway', 
                    value_name='expression')

    df_long['pathway'] = pd.Categorical(df_long['pathway'], 
                                        categories=pec_mito_mean_df.columns, 
                                        ordered=True)
    df_r = pandas2ri.py2rpy(df_long)

    plot_energy_loess(df_r, 
                    x=x_axis, 
                    path_fig=path_fig, 
                    colors=['#F15B14', '#698EC9'],
                    filename=f"both_mito_dev_loess_{x_axis}_ctx_final.svg",
                    width=12, height=9)


##########################
# dev related traejctories
# load cell type genes
kang_sets = pd.read_csv(path_data+'kang2011_genesets.csv')
kang_genes = kang_sets.groupby('Functional group')['Gene symbol'].apply(list).reset_index()
kang_genes_dict = dict(zip(kang_genes['Functional group'], kang_genes['Gene symbol']))

li_sets = pd.read_csv(path_data+'li2018_genesets.csv')
li_genes = li_sets.groupby('Cell type')['Gene symbol'].apply(list).reset_index()
li_genes_dict = dict(zip(li_genes['Cell type'], li_genes['Gene symbol']))

my_cell_dict = {
                'neuron_diff': ['MAP1B', 'MAP2', 'TUBB'],
                'synapse_dev': ['SYP', 'SYPL1', 'SYPL2', 'SYN1'],
                'neuron_migr': ['DCX'],
                'dendrite_dev': ['MAP1A', 'MAPT', 'CAMK2A'],
                'axon_dev': ['CNTN2']}

# pu the three dictionaries together
all_gene_sets = {**kang_genes_dict, **li_genes_dict, **my_cell_dict}

# make expression matrices
pec_cell_exp = {}
pec_cell_mean = {}
for key, value in all_gene_sets.items():
    pec_cell_exp[key] = pec_dev_exp.loc[:, pec_dev_exp.columns.isin(value)]
    pec_cell_mean[key] = np.mean(pec_cell_exp[key], axis=1)

pec_cell_mean_df = pd.DataFrame(pec_cell_mean)
df_cell = pd.concat([pec_dev_sample_info, pec_cell_mean_df], axis=1)

for x_axis in ['log_age_days']:

    df_cell_long = pd.melt(df_cell[df_cell['NCXRegion'] == 'NCX'], 
                        id_vars=[x_axis, 'Species'],
                        value_vars=pec_cell_mean_df.columns,
                        var_name='pathway', 
                        value_name='expression')

    df_cell_long['pathway'] = pd.Categorical(df_cell_long['pathway'], 
                                            categories=pec_cell_mean_df.columns, 
                                            ordered=True)
    df_cell_r = pandas2ri.py2rpy(df_cell_long)

    plot_energy_loess(df_cell_r,
                        x=x_axis,
                        color='Species',
                        path_fig=path_fig,
                        filename=f"both_cell_dev_loess_{x_axis}_ctx.svg",
                        width=14, height=10)


##############################################
# make latex table of genes in energy pathways
final_genes = {}
for pathway, value in pec_energy_exp.items():
    final_genes[pathway] = value.columns

# write ot latex table
final_genes = {}
for pathway, df in pec_energy_exp.items():
    final_genes[pathway] = ', '.join(sorted(df.columns))

table = pd.DataFrame(final_genes.items(), columns=['Pathway', 'Genes'])

pd.set_option('display.max_colwidth', None)
table.to_latex(path_result + 'pec_energy_genes_table.tex', 
               index=False, label="tab:energy_genes")

# for both energy and mito pathwyas
# write the final gene sets in exp to xlsx file
with pd.ExcelWriter(path_result + 'pec_energy_mito_genes.xlsx') as writer:
    for key, exp_dict in [('energy', pec_energy_exp), ('mitocarta', pec_mitocarta_exp)]:
        df = pd.DataFrame([(pathway, ', '.join(sorted(d.columns))) 
                           for pathway, d in exp_dict.items()],
                          columns=['pathway', 'genes'])
        df.to_excel(writer, sheet_name=key, index=False)