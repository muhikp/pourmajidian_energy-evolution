
'''
Replication of human MitoCarta trajectories
using brainspan microarray data

1. clean up
2. plot trajectories

author: moohebat
'''

# import packages
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.utils import plot_energy_loess_hbt

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

# data was downloaded from brainspan "Exon microarray summarized to genes"
# expression matrix
bs_exp = pd.read_csv(path_data + 'expression_matrix.csv', header=None) # (gene x sample)
bs_exp = bs_exp.iloc[:, 1:] # (17604 x 492)

# gene metadata
gene_data = pd.read_csv(path_data + 'rows_metadata.csv')['gene_symbol']

# sample metadata
sample_data = pd.read_csv(path_data + 'columns_metadata.csv').set_index('column_num')

#simple stats
np.unique(sample_data['donor_id']).shape #35 donors
np.unique(sample_data['age']).shape #27 developmental stages
np.unique(sample_data['structure_acronym']).shape #26 structure acronyms

order = ['8 pcw', '9 pcw', '12 pcw', '13 pcw', 
         '16 pcw', '17 pcw',  '19 pcw', '21 pcw', 
         '24 pcw', '25 pcw', '26 pcw','4 mos', '10 mos', 
         '1 yrs', '2 yrs', '3 yrs', '4 yrs', '8 yrs',
         '13 yrs', '15 yrs','18 yrs','21 yrs', '23 yrs', 
         '30 yrs', '36 yrs', '37 yrs', '40 yrs', ]

# adding age category
def map_to_category(list_names, list_items):
    map_to_category = {item: category for category, items in \
                       zip(list_names, list_items) for item in items}
    return map_to_category

# broad age category Li et al 2018
fetal = ['8 pcw', '9 pcw', '12 pcw', '13 pcw', '16 pcw', 
         '17 pcw', '19 pcw', '21 pcw','24 pcw', '25 pcw', 
         '26 pcw', '35 pcw', '37 pcw']
infant = ['4 mos','10 mos','1 yrs']
child = ['2 yrs', '3 yrs','4 yrs','8 yrs', '11 yrs']
adolescent = ['13 yrs', '15 yrs', '18 yrs', '19 yrs']
adult = ['21 yrs', '23 yrs','30 yrs', '36 yrs', '37 yrs',
        '40 yrs']

groups = [fetal,infant,child,adolescent,adult]
names = ['fetal','infant','child','adolescent','adult']
# map
sample_data['age_group'] = sample_data['age'].map(map_to_category(names, groups))

# adding cortex, non_cortex division
regions = np.unique(sample_data['structure_name']) # 26
# cortex and subcoretx divisions
cortex = []
subcortex = []
for region in regions:
    if 'cortex' in region:
        cortex.append(region)
    else:
        subcortex.append(region)

cortex.remove('cerebellar cortex')
subcortex.append('cerebellar cortex')
# there are 15 cortical and 11 subcortical regions

names = ['cortex', 'subcortex']
ctx = [cortex, subcortex]
# map
sample_data['ctx'] = sample_data['structure_name'].map(map_to_category(names, ctx))

# convert age to postconception days
sample_data['age_days'] = np.nan

for i, age in enumerate(sample_data['age']):
    if 'pcw' in age:
        sample_data.loc[i+1, 'age_days'] = float(age.split(' ')[0]) * 7
    elif 'mos' in age:
        sample_data.loc[i+1, 'age_days'] = 40 * 7 + float(age.split(' ')[0]) * 30.43
    elif 'yrs' in age:
        sample_data.loc[i+1, 'age_days'] = 40 * 7 + float(age.split(' ')[0]) * 12 * 30.43


###########
# QC
# sample data cleanup
# region qc
# focusing on cortical regions
sample_data2 = sample_data[sample_data['ctx'] == 'cortex'] #345
# sample_data2 = sample_data
# 1. keep regions with at least 1 sample in each age group
np.unique(sample_data2['structure_acronym']).shape  # 26 unique regions
groups = sample_data2.groupby(['structure_acronym', 
                              'age_group']).size().unstack(fill_value=0)

structures = groups[(groups >= 1).all(axis=1)].index
sample_data2 = sample_data2[sample_data2['structure_acronym'].isin(structures)]
dropped = list(set(sample_data['structure_acronym']) - set(sample_data2['structure_acronym']))
# dropped ['M1C-S1C', 'CGE', 'MD', 'Ocx', 'CBC', 'AMY', 
# 'TCx', 'DTH', 'LGE', 'HIP', 'PCx', 'STR', 'MGE', 'URL', 'CB']
# 334 samples left

bs_exp = bs_exp.loc[:, sample_data2.index]
# 17604 x 334

##############
# gene cleanup
# 1. drop duplicate genes
gene_uniq = gene_data.drop_duplicates()
bs_exp = bs_exp.iloc[gene_uniq.index, :] # (17282 rows x 334 columns)

# 2. drop non-expressed genes
# excluded genes with a log2-transformed expression value <6
bs_exp = bs_exp.loc[(bs_exp >= 6).any(axis=1)] # 13635

gene_qc = gene_uniq.loc[bs_exp.index]

bs_df = pd.concat([gene_qc, bs_exp], axis=1)

# save the final expression matrix
sample_data2.to_csv(path_result+'sample_data_micro_qc.csv')
bs_df.to_csv(path_result+'brainspan_micro_exp_qc.csv')


# quartile normalize
# load data
sample_data = pd.read_csv(path_result+'sample_data_micro_qc.csv').set_index('column_num')
bs_df = pd.read_csv(path_result+'brainspan_micro_exp_qc.csv').iloc[:, 1:]

df = pd.concat([sample_data['donor_id'].reset_index(drop=True), 
                bs_df.set_index('gene_symbol').T.reset_index(drop=True)], 
                axis=1)

percentile_75 = df.groupby('donor_id').apply(lambda group: np.percentile(group.iloc[:, 1:].values.flatten(), 75))
mean_p75 = np.mean(percentile_75)

df_norm = pd.DataFrame()
for donor in np.unique(sample_data['donor_id']):
    group = df.groupby('donor_id').get_group(donor)
    percent75 = np.percentile(group.iloc[:, 1:].values.flatten(), 75)
    group.iloc[:, 1:] = (group.iloc[:, 1:] / percent75) * mean_p75
    df_norm = pd.concat([df_norm, group], axis=0)

df_norm = df_norm.sort_index()

# save
df_norm.to_csv(path_result+'brianspan_exp_micro_uqnorm.csv')



'''
lifespan trajectory
'''

# load mitocartsa dict
with open(path_result+'mitocarta_dict.pickle', 'rb') as f:
    mitocarta_dict = pickle.load(f)

# read brainspan normalzied data
sample_data = pd.read_csv(path_data+'brainspan_sample_info_micro_qc.csv').set_index('column_num')
df_norm = pd.read_csv(path_data+'brainspan_exp_micro_uqnorm.csv').iloc[:, 1:]

final_df = df_norm.drop('donor_id', axis=1).T.reset_index(names='gene_symbol')

# get mitocarta expression and mean
bs_exp_mitocarta = {}
bs_mean_mitocarta = {}
for key, value in mitocarta_dict.items():
        bs_exp_mitocarta[key] = final_df[final_df.gene_symbol.isin(value)]
        bs_mean_mitocarta[key] = np.mean(bs_exp_mitocarta[key].iloc[:, 1:], axis=0)

bs_mean_mitocarta_df = (pd.DataFrame.from_dict(bs_mean_mitocarta))

# keep important pathways for plotting
imp_mito = pd.read_csv(path_data+'imp_mito_plotting.csv', header=None).iloc[:,0].tolist()
bs_mean_mitocarta_df= bs_mean_mitocarta_df[imp_mito]

# add sample info for plotting
df_mitocarta = pd.concat([sample_data.reset_index(drop=True), bs_mean_mitocarta_df], axis=1)
df_mitocarta['log_age_days'] = np.log10(df_mitocarta['age_days'].values.astype('float'))
df_mitocarta['Species'] = 'Human'

# smooth
for x_axis in ['log_age_days']:

    df_long = pd.melt(df_mitocarta, 
                    id_vars=[x_axis, 'Species'],
                    value_vars=bs_mean_mitocarta_df.columns,
                    var_name='pathway', 
                    value_name='expression')

    df_long['pathway'] = pd.Categorical(df_long['pathway'], 
                                        categories=bs_mean_mitocarta_df.columns, 
                                        ordered=True)
    df_r = pandas2ri.py2rpy(df_long)

    plot_energy_loess_hbt(df_r, 
                    x=x_axis, 
                    width=10,
                    height=7,
                    path_fig=path_fig, 
                    filename=f"mitocarta_dev_loess_{x_axis}_brainspan_micro_final.svg")