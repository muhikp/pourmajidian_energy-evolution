
'''
functions used throughout the analyses
Author: Moohebat
Date: 16/04/2026
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import spearmanr
import abagen
from netneurotools import datasets
from netneurotools.freesurfer import parcels_to_vertices
from neuromaps.datasets import fetch_fsaverage
from nilearn.datasets import fetch_atlas_schaefer_2018
from surfplot import Plot

import rpy2.robjects as robjects
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects.lib import grdevices
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# boxplots for lifespan trajectories
def plot_energy_trajectories(df, species, category, columns,
                              figsize, rotations,
                              outpath, filename):
    
    df = df.copy()
    species_df = df[df['Species'] == species]
    
    n = len(columns)
    ncols = 5
    nrows = math.ceil(n / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                                sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, key in zip(axes, columns):
        sns.boxplot(data=species_df, x=category, y=key, ax=ax,
            boxprops={'facecolor': 'none', 'edgecolor': 'darkgrey'},
            whiskerprops={'color': 'darkgrey'},
            capprops={'visible': False},
            medianprops={'color': 'crimson'},
            showfliers=False, linewidth=0.7, width=0.4)
        
        sns.stripplot(data=species_df, x=category, y=key, ax=ax,
            color='sandybrown', size=3, alpha=0.4, jitter=True, zorder=-5)
        
        sns.lineplot(data=species_df, x=category, y=key, ax=ax,
            estimator=np.median, errorbar=None, color='mediumvioletred',
            marker='o', markersize=0, markeredgewidth=0, linewidth=0.7)
        
        ax.set_title(key)
        ax.set_xlabel(None)
        ax.set_ylabel('normalized expression' if ax == axes[0] else None)
        sns.despine(ax=ax)
        plt.setp(ax.xaxis.get_majorticklabels(), 
                    rotation=rotations,
                    ha='right', rotation_mode='anchor')

    for ax in axes[n:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.savefig(outpath + filename)
    plt.show()
    

def plot_energy_loess(df, x, vlines=None, 
                      color='Species', colors=['orange', 'lightskyblue'], 
                      brewer_palette=None,
                      method='loess', 
                      nrow=None,
                      path_fig=None, filename=None,
                      width=10, height=9):
    
    pp = (ggplot2.ggplot(df) +
          ggplot2.aes_string(x=x, y='expression', color=color) +
          ggplot2.geom_point(size=1.4, stroke=0, alpha=0.15) +
          ggplot2.geom_smooth(method=method, 
                              se=False, size=0.6) +
          ggplot2.facet_wrap("~pathway", scales='free', **({'nrow': nrow} if nrow else {})) +
          ggplot2.scale_color_manual(values=colors) +
          (ggplot2.scale_color_brewer(palette=brewer_palette) if brewer_palette 
           else ggplot2.scale_color_manual(values=colors)) +
          ggplot2.theme(axis_line=ggplot2.element_line(color='black'),
                        panel_background=ggplot2.element_blank(),
                        panel_grid_major=ggplot2.element_blank(),
                        panel_grid_minor=ggplot2.element_blank(),
                        strip_background=ggplot2.element_blank()) + 
            ggplot2.geom_vline(xintercept=np.log10(164),
                                color='grey', linetype='solid', 
                                size=0.7, show_legend=False) +
            ggplot2.geom_vline(xintercept=np.log10(266),
                                color='dimgrey', linetype='solid', 
                                size=0.7, show_legend=False))
    if path_fig and filename:
        r(f'svglite::svglite("{path_fig + filename}", width={width}, height={height})')
    
    pp.plot()
    
    if path_fig and filename:
        grdevices.dev_off()


def plot_traj_regions_boxes(df, columns, region_col='region',
                                     regions=None, colors=None,
                                     box=False,
                                     ncols=5, figsize=(9,7),
                                     outpath=None, filename=None):
    import math
    regions = regions or df[region_col].unique()

    n = len(columns)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=False)
    axes = axes.flatten()

    for ax, key in zip(axes, columns):
        for region, color in zip(regions, colors):
            sub = df[df[region_col] == region]

            sns.stripplot(data=sub, x='age', y=key, ax=ax,
                color=color, size=3, alpha=0.4, jitter=False, zorder=-5)

            if box:
            # boxplot
                sns.boxplot(data=sub, x='age', y=key, ax=ax,
                    boxprops={'facecolor': 'none', 'edgecolor': color},
                    whiskerprops={'color': color},
                    capprops={'visible': False},
                    medianprops={'color': color},
                    showfliers=False, linewidth=0.7, width=0.4)
                for patch in ax.patches[-len(sub['age'].unique()):]:
                    patch.set_alpha(0.5)

            sns.lineplot(data=sub, x='age', y=key, ax=ax,
                estimator='median',
                errorbar=None,
                color=color, linewidth=0.9,
                marker='o', markersize=0, markeredgewidth=0,
                alpha=0.7)  # or 'sd' for standard deviation
        
        ax.set_title(key, fontsize=7)    
        ax.set_xlabel(None)
        ax.set_ylabel('normalized expression' if ax == axes[0] else None)
        sns.despine(ax=ax)
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=90, ha='center', rotation_mode='anchor')

    handles = [plt.Line2D([0],[0], color=c, lw=1.5, label=r)
               for r, c in zip(regions, colors)]
    fig.legend(handles=handles, loc='lower right', fontsize=7, frameon=False)

    for ax in axes[n:]:
        fig.delaxes(ax)
    plt.tight_layout()
    if outpath and filename:
        plt.savefig(outpath + filename, bbox_inches='tight')


def plot_energy_loess_cardoso(df, x='log_age_days', 
                              x_labels=None,
                              vlines=None,
                              regions=None,
                              colors=None,
                              method='loess', 
                              width=10, height=9,
                              path_fig=None, filename=None):
    ages = np.sort(np.array(df.rx2('log_age_days')))
    x_breaks = np.unique(ages)
    if x_labels:
        x_labels = robjects.StrVector(x_labels)
    
    if regions and colors:
        geom_pt = ggplot2.geom_point(size=1.5, stroke=0, alpha=0.4)
        geom_sm = ggplot2.geom_smooth(method=method, se=False, size=0.5)
        aes = ggplot2.aes_string(x=x, y='expression', color='region')
        color_scale = ggplot2.scale_color_manual(values=robjects.StrVector(colors),
                                                 breaks=robjects.StrVector(regions))
    else:
        geom_pt = ggplot2.geom_point(size=1.5, stroke=0, alpha=0.4, color='sandybrown')
        geom_sm = ggplot2.geom_smooth(method=method, se=False, size=0.5, color='mediumvioletred')
        aes = ggplot2.aes_string(x=x, y='expression')
        color_scale = None

    pp = (ggplot2.ggplot(df) +
          aes + geom_pt + geom_sm +
          ggplot2.facet_wrap("~pathway", scales='free') +
          ggplot2.theme(axis_line=ggplot2.element_line(color='black'),
                        panel_background=ggplot2.element_blank(),
                        panel_grid_major=ggplot2.element_blank(),
                        panel_grid_minor=ggplot2.element_blank(),
                        strip_background=ggplot2.element_blank(),
                        axis_text_x=ggplot2.element_text(angle=90, 
                                                         vjust=0.5, 
                                                         hjust=1,
                                                         size=5),
                        axis_text_y=ggplot2.element_text(size=5)) +
          ggplot2.scale_x_continuous(
              breaks=robjects.FloatVector(x_breaks),
              labels=x_labels)
          )
    
    if color_scale:
        pp = pp + color_scale
    if vlines:
        for vline in vlines:
            pp = pp + ggplot2.geom_vline(xintercept=np.log10(vline),
                                         color='grey', linetype='solid',
                                         size=0.7, show_legend=False)
    
    if path_fig and filename:
        r(f'svglite::svglite("{path_fig + filename}", width={width}, height={height})')
    pp.plot()
    if path_fig and filename:
        grdevices.dev_off()


def plot_energy_loess_hbt(df, x, vlines=None, 
                      method='loess', 
                      path_fig=None, filename=None,
                      width=10, height=9):
    
    pp = (ggplot2.ggplot(df) +
          ggplot2.aes_string(x=x, y='expression') +
          ggplot2.geom_point(size=1.4, stroke=0, alpha=0.3, color='sandybrown') +
          ggplot2.geom_smooth(method=method, 
                              se=False, size=0.5, color='mediumvioletred') +
          ggplot2.facet_wrap("~pathway", scales='free') +
          ggplot2.scale_color_manual() +
          ggplot2.theme(axis_line=ggplot2.element_line(color='black'),
                        panel_background=ggplot2.element_blank(),
                        panel_grid_major=ggplot2.element_blank(),
                        panel_grid_minor=ggplot2.element_blank(),
                        strip_background=ggplot2.element_blank()) + 
            ggplot2.geom_vline(xintercept=np.log10(280),
                                color='grey', linetype='solid', 
                                size=0.7, show_legend=False))
    if path_fig and filename:
        r(f'svglite::svglite("{path_fig + filename}", width={width}, height={height})')
    
    pp.plot()
    
    if path_fig and filename:
        grdevices.dev_off()


def load_expression(scale):

    '''get expression dictionary from abagen'''

    schaefer = fetch_atlas_schaefer_2018(n_rois=scale) # 7networks and 1mm res.

    expression = abagen.get_expression_data(schaefer['maps'],
                                            lr_mirror='bidirectional', 
                                            missing='interpolate', 
                                            return_donors=True)
    return expression


def filter_expression_ds(expression, ds):
    '''
    filter expression matrix based on given differential stability value
    and average across donors
    '''
    
    # keeping stable genes across donors and samples.
    rexpression, diff_stability = abagen.correct.keep_stable_genes(list(expression.values()), 
                                                           threshold=ds, 
                                                           percentile=False, 
                                                           return_stability=True)
    # rexpression is a type list of dataframes
    # rexpression is still by donor, so we are avergaing over donors
    expression_ds = pd.concat(rexpression).groupby('label').mean()

    return expression_ds, diff_stability


def geneset_expression(expression, gene_list, filename, outpath, save=False):

    '''
    filters ahba expression data for a given gene set
    '''
    # getting the expression values for a given gene list
    filtered_expression = expression[expression.columns.intersection(gene_list)]
    # save
    if save:
        filtered_expression.to_csv(outpath+filename+'_exp.csv')

    return filtered_expression



def plot_schaefer_fsaverage(data, hemi=None, cmap = 'plasma', resolution=400, cbar_range=None):
    '''
    function to plot parcellated schaefer data onto fsaverage surface
    uses surfplot
    '''

    # load schaefer atlas
    if resolution == 400:
        scale = '400Parcels7Networks'
    elif resolution == 100:
        scale = '100Parcels7Networks'
    
    schaefer = datasets.fetch_schaefer2018('fsaverage')[scale]
    # convert parcellated data into vertex-wise data
    x = parcels_to_vertices(data, lhannot=schaefer.lh, rhannot=schaefer.rh)
    if hemi == 'L':
        #keeping only the left hemisphere data
        x = x[:len(x)//2]
    if hemi == 'R':
        x = x[len(x)//2:]
    
    # load 
    surfaces = fetch_fsaverage(density='164k')
    # sulc_lh, sulc_rh = surfaces['sulc']
    lh, rh = surfaces['inflated']
    # lh_sulc, rh_sulc = surfaces['sulc']
    p = Plot(surf_lh=lh, surf_rh=rh, zoom=1.5, brightness=0.8)
    # if only the left hemisphere
    if hemi == 'L':
        p = Plot(surf_lh=lh, zoom=1.2)
    if hemi == 'R':
        p = Plot(surf_rh=rh, zoom=1.2)
    # add data layer to surface    
    p.add_layer(x, cmap=cmap, cbar=True, color_range=(cbar_range))

# # for only left hemisphere
#     if hemi == 'L':
#         p = Plot(surf_lh=lh, zoom=1.2)
#         p.add_layer(x[:163842], cmap=cmap, cbar=True)
#     # color_range = (min(x), max(x))

    # p.add_layer({'left': sulc_lh, 'right': sulc_rh}, cmap='binary_r', cbar=False, )
    # kws = {'fontsize': 8, 'draw_border': False}

    kws = {'fontsize': 8, 'n_ticks': 2, 'shrink': 0.4, 'aspect': 10 ,'draw_border': False}
    fig = p.build(figsize=(3,3), cbar_kws= kws)
    plt.tight_layout()
    # fig.show()



def corr_spin_test(data, map, spins, 
                   scattercolor='sandybrown', 
                   linecolor='grey', 
                   plot=False,
                   ax=None):

    # generating spins and calculating corr distribution
    nspins = spins.shape[1]
    corr_null = np.zeros((spins.shape[1],))
    corr, _ = spearmanr(data, map)
    for i in range(spins.shape[1]):
        corr_null[i], _ = spearmanr(data[spins[:, i]], map)

    # calculating p_spin
    p_spin = (
        1
        + np.sum(abs(corr_null - np.mean(corr_null)) >= abs(corr - np.mean(corr_null)))
    ) / (nspins + 1)

    if plot:
        # plotting correlation plot and reporting pval
        # scatter plot
        plt.figure(figsize=(3, 3))
        sns.regplot(
            x=data,
            y=map,
            scatter=True,
            fit_reg=True,
            color=scattercolor,
            ci=None,
            scatter_kws={"linewidth": 0.3, "s": 12, "alpha": 1, 'edgecolor': 'grey'},
            line_kws={"color": linecolor, "lw": 0, "alpha": 0.8},
            ax=ax
        )
        sns.despine()
        plt.title("r ={:.2f}".format(corr) + ",  p_spin ={:.2e}".format(p_spin))
        plt.tight_layout()
        # plt.show()
    return corr, corr_null, p_spin



def pair_corr_spin(x, y, spins):
    corr_df = pd.DataFrame(columns=x.columns, index=y.columns, dtype=np.float64)
    pspin_df = pd.DataFrame(columns=x.columns, index=y.columns, dtype=np.float64)
    for col in x.columns:
        for col2 in y.columns:
            corr_df.loc[col2, col], _, pspin_df.loc[col2, col] = corr_spin_test(np.array(x[col]),
                                                                                      y[col2], 
                                                                                      spins,
                                                                                      plot=False)
    return corr_df, pspin_df


