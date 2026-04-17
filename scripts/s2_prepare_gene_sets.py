
'''
script to prepare energy gene sets for analysis
consolidate shared pathways between my gene sets (GO and reactome)
with the mitocarta gene sets

author: moohebat
'''

# importy import
import pandas as pd
import pickle

# path to directories
path_data = './data/'
path_result = './results/'
path_fig = './figures/'

# load energy gene sets
with open(path_result+'energy_genelist_dict.pickle', 'rb') as f:
    energy_dict = pickle.load(f)

# load mitocarta 3.0 gene list
mito_pathway = pd.read_csv(path_data + 'mitocarta_pathways.csv')

# convert to dict
mitocarta_dict = dict(zip(mito_pathway['MitoPathway'], 
                            mito_pathway['Genes'].str.split(', ')))


# consolidate shared gene sets between my pathways and mitocarta
mine_mito_map = {'tca': 'TCA cycle',
                 'oxphos': 'OXPHOS',
                 'complex1': 'Complex I', 
                 'complex2': 'Complex II', 
                 'complex3': 'Complex III', 
                 'complex4': 'Complex IV',
                 'atpsynth': 'Complex V',
                 'mas': 'Malate-aspartate shuttle',
                 'gps': 'Glycerol phosphate shuttle',
                 'creatine': 'Creatine metabolism',
                 'ros_detox': 'ROS and glutathione metabolism',
                 'bcaa_cat': 'Branched-chain amino acid metabolism',
                 }

genes_updated = energy_dict.copy()
for my_pathway, mito_pathway in mine_mito_map.items():
    genes_updated[my_pathway] = list(set(energy_dict[my_pathway]).union(set(mitocarta_dict[mito_pathway])))

# save updated main energy gene sets
with open(path_result+'energy_genelist_dict_consolidated.pickle', 'wb') as f:
    pickle.dump(genes_updated, f)

# save mitocarta gene sets
with open(path_result+'mitocarta_dict.pickle', 'wb') as f:
    pickle.dump(mitocarta_dict, f)