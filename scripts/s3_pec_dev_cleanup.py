
'''
Script to clean psychencode developmental 
macaque and human lifespan data,

author: moohebat
'''

# importy import
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 8})

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load expression matrix and sample info
pec_dev_exp = pd.read_csv(path_data + 
                                 'mac_human_dev_exp.csv')

pec_dev_sample_info = pd.read_csv(path_data + 
                                             'mac_human_dev_sample_info.csv')

# function to map age to age group
def map_to_category(list_names, list_items):
    map_to_category = {item: category for category, items in \
                       zip(list_names, list_items) for item in items}
    return map_to_category

################
# 1. sample info

# 1.1. add age group column
pec_dev_sample_info.loc[pec_dev_sample_info['Species']=='Macaque']['Age'].unique().shape
# ['E60', 'E81', 'E82', 'E80', 'E111', 'E110', 'P2', 'P0', '1Y', '7M',
#  '2Y', '4Y', '7Y', '5Y', '11Y', '3.5Y']
# macaque has 16 ages

pec_dev_sample_info.loc[pec_dev_sample_info['Species']=='Human']['Age'].unique().shape
# ['8 PCW', '9 PCW', '12 PCW', '13 PCW', '16 PCW', '17 PCW', '19 PCW',
#  '21 PCW', '22 PCW', '37 PCW', '4 M', '10 M', '1 Y', '3 Y', '4 Y',
#  '8 Y', '11 Y', '13 Y', '15 Y', '19 Y', '21 Y', '23 Y', '30 Y',
#  '36 Y', '37 Y', '40 Y']
# human has 26 ages

mac_age_order = ['E60', 'E80', 'E81', 'E82', 'E110', 'E111', 
                 'P0', 'P2', '7M', '1Y', '2Y', '3.5Y', '4Y', '5Y',
                 '7Y', '11Y']

human_age_order = ['8 PCW', '9 PCW', '12 PCW', '13 PCW', '16 PCW', 
                   '17 PCW', '19 PCW', '21 PCW', '22 PCW', '37 PCW',
                   '4 M', '10 M', '1 Y', '3 Y', '4 Y', '8 Y', 
                   '11 Y', '13 Y', '15 Y', '19 Y', 
                   '21 Y', '23 Y', '30 Y', '36 Y', '37 Y', '40 Y']

# age groups
prenate = ['E60', 'E80', 'E81', 'E82', 'E110', 'E111',
         '8 PCW', '9 PCW', '12 PCW', '13 PCW', 
         '16 PCW', '17 PCW', '19 PCW',
         '21 PCW', '22 PCW', '37 PCW']
infant = ['P0', 'P2', '7M',
          '4 M', '10 M', '1 Y']
child = ['1Y', '2Y',
         '3 Y', '4 Y', '8 Y', '11 Y']
adolescent = ['3.5Y', '4Y',
              '13 Y', '15 Y', '19 Y']
adult = ['5Y', '7Y', '11Y',
         '21 Y', '23 Y', '30 Y', '36 Y', '37 Y', '40 Y']

# map
groups1 = [prenate,infant,child,adolescent,adult]
names1 = ['prenate','infant','child','adolescent','adult']
pec_dev_sample_info['age_group'] = pec_dev_sample_info['Age'].map(map_to_category(names1, groups1))

# broad age category kang et al 2011, only for human
early_fetal = ['8 PCW', '9 PCW', '12 PCW']
mid_fetal = ['13 PCW', '16 PCW', '17 PCW', '19 PCW', '21 PCW', '22 PCW']
late_fetal = ['37 PCW']
infant = ['4 M', '10 M', '1 Y']
early_child = ['3 Y', '4 Y']
late_child = ['8 Y', '11 Y']
adolescent = ['13 Y', '15 Y', '19 Y']
adult = ['21 Y', '23 Y', '30 Y', '36 Y', '37 Y', '40 Y']

# map
names2 = ['early_fetal','mid_fetal','late_fetal','infant', 
         'early_child','late_child','adolescent','adult']
group2 = [early_fetal, mid_fetal, late_fetal, infant,
          early_child, late_child, adolescent, adult]
pec_dev_sample_info['age_group2'] = pec_dev_sample_info['Age'].map(map_to_category(names2, group2))


# add log(age) column
pec_dev_sample_info['log_age_days'] = np.log10(pec_dev_sample_info['Days'])
pec_dev_sample_info['log_age_days_predicted'] = np.log10(pec_dev_sample_info['Predicted age (PC Days)'])


# add anatomical division column
front = ['MFC', 'DFC', 'OFC', 'VFC', 'M1C']
par = ['PC', 'IPC', 'S1C',]
temp = ['TC', 'STC', 'ITC', 'A1C']
occ = ['OC', 'V1C']
lim = ['AMY', 'HIP']
groups3 = [front,par,temp,occ,lim]
names3 = ['front','par','temp','occ','lim']
pec_dev_sample_info['network'] = pec_dev_sample_info['Region'].map(map_to_category(names3, groups3))

# save new sample info
pec_dev_sample_info.to_csv(path_result + 'pec_dev_sample_info_qc.csv', 
                             index=True)


###############
# basic cleanup

# 1. genes
# fix probe ID column, cause it has both symbol and ensembl id
pec_dev_exp['ProbeID'] = pec_dev_exp['ProbeID'].str.split('|').str[1]
pec_dev_exp.rename(columns={'ProbeID': 'gene_symbol'}, inplace=True)
# 27932 genes by 826 samples

# !important: sample info and expression data are not in the same order
# im gonna order it based on sample_info
pec_dev_exp1 = pec_dev_exp.set_index('gene_symbol').T

# reindex to match sample info
pec_dev_exp2 = pec_dev_exp1.reindex(pec_dev_sample_info['Sample'])
pec_dev_exp2.reset_index(inplace=True, drop=True)
# 826 samples
# 27,932 genes

# i'm doing the gene qc on the mac+human matrix, 
# cause i need the genes to be the same
# 1.1. drop duplicate genes
pec_dev_exp3 = pec_dev_exp2.T.reset_index()
pec_dev_exp_uniq = pec_dev_exp3.drop_duplicates(subset='gene_symbol', 
                                                    keep=False)
# 27,625 unique genes

# 1.2. keep "expressed" genes
# # check gene expression distribution in each sample
# for column in pec_dev_exp_uniq.iloc[:, 1:].columns:
#     plt.hist(pec_dev_exp_uniq[column], bins=100, label=column)
#     plt.show()
# # pretty skewed towards 0, and is obviously log2 transformed

# keep genes that have log2(rpkm) >= 1 in half the samples
pec_dev_exp4 = pec_dev_exp_uniq[(pec_dev_exp_uniq.iloc[:, 1:] >= 1).mean(axis=1) >= 0.5]
# 12,509 genes remain

# # check gene expression distribution again
# for column in pec_dev_exp4.iloc[:, 1:].columns:
#     plt.hist(pec_dev_exp4[column], bins=100, label=column)
#     plt.show()
# # distribution looks more normal now, skewed to right, which makes sense

# save expression matrix
pec_dev_exp5 = pec_dev_exp4.set_index('gene_symbol').T
pec_dev_exp5.to_csv(path_result + 'pec_dev_exp_qc.csv', index=True)

# 3. upper quartile normalization
pec_dev_exp = pd.read_csv(path_result + 'pec_dev_exp_qc.csv', index_col=0)
pec_dev_sample_info = pd.read_csv(path_result + 'pec_dev_sample_info_qc.csv', index_col=0)

df = pd.concat([pec_dev_sample_info[['Brain']], 
                pec_dev_exp],
                axis=1)

percentile_75 = df.groupby('Brain').apply(
    lambda group: np.percentile(group.values.flatten(), 75))
mean_p75 = np.mean(percentile_75)

scale_factors = pec_dev_sample_info['Brain'].map(percentile_75)

mac_human_exp_uqnorm = pec_dev_exp.div(scale_factors.values, axis=0) * mean_p75

# save
mac_human_exp_uqnorm.to_csv(path_result + 'pec_dev_exp_uqnorm.csv')


#################
# some data stats

pec_dev_sample_info = pd.read_csv(path_result + 'pec_dev_sample_info_qc.csv', index_col=0)

# set all ages as categorical and ordered
# keeps the order later for plotting
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

# number of samples for each region in each age
# only cortical
pec_dev_sample_info = pec_dev_sample_info[pec_dev_sample_info['NCXRegion'] == 'NCX']
region_age_sample_count = pec_dev_sample_info.groupby(['Region', 'Age']).size().unstack(fill_value=0)

plt.figure(figsize=(9, 4))
sns.heatmap(
    data=region_age_sample_count,
    annot=True,
    cmap='OrRd',
    linewidths=0,
    cbar=False
)
plt.title('number of samples by age and region')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(path_fig + 'pec_dev_sample_count_by_age_region.svg', dpi=300)
plt.show()