

# get human orthologs for macaque, rat, mouse, chicken

# date 28/02/2025
# author moohebat

###########################
# 1. using alleninf package

install.packages("R.utils")

install.packages("remotes", repos='http://cran.us.r-project.org')
remotes::install_github("AllenInstitute/GeneOrthology")

library(GeneOrthology)
library(R.utils)

taxIDs <- setNames(c(9606,10090,9541),
                   c("human","mus musculus","macaca fascicularis"))

build_orthology_table(taxIDs = taxIDs,  primaryTaxID = 9606, 
                      outputFilePrefix="mus_musculus_macaque_fasci_human_orthologs", )


# tax ids were retrieved from ncbi taxonomy browser
taxIDs <- setNames(c(9606, 10090, 9544, 13616, 9986, 10116, 9031),
                   c("human", "mouse", "macaque", "opossum", "rabbit", "rat", "chicken"))

options(timeout = 1200) 

build_orthology_table(taxIDs = taxIDs,  
                      primaryTaxID = 9606, 
                      addEnsemblID = TRUE,
                      includeNonMammalianSpecies = TRUE,
                      outputFilePrefix="human_orthologs_table", )


