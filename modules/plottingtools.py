from matplotlib.pyplot import *
from matplotlib.colors import LinearSegmentedColormap
from numpy import *
import rc_parameters

def my_figure(figsize = (2,2)):
    fig, ax = subplots(figsize = figsize, dpi=300)
    fmt = matplotlib.ticker.StrMethodFormatter("{x}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    return fig,ax

def my_fill_between(x, F, col, colfill, labels,  **pars):
    ls = pars['ls']
    lw = pars['lw']
    ms = pars['markersize']
    m  = [nanmean(f,0) for f in F]
    s  = [nanstd(f,0) for f in F]
    ntrials = F[0].shape[0]
    a = sqrt(ntrials) if pars['err'] == 'se' else 1
    for i in range(len(m)):
        plot(x, m[i], ls, lw = lw, color = col[i], label = labels[i],markersize=ms)
        fill_between(x, m[i]-s[i]/a, m[i]+s[i]/a,color = colfill[i])

def my_boxplot(figsize, data, labels, rotation, facecolor, colorwhisk, colorcaps, colorfliers, width):
    fig = figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    flierprops      = dict(marker = 'o', markerfacecolor = colorfliers, markersize = 3, markeredgewidth = 0, linestyle = 'none')
    boxprops        = dict(linewidth = 1.)
    capprops        = dict(linewidth = 1.)
    bp = ax.boxplot(data, flierprops = flierprops, widths = width, patch_artist = True,
                     boxprops = boxprops, capprops = capprops)
    for i,box in enumerate(bp['boxes']):
        box.set(facecolor=facecolor[i], color=facecolor[i])
    for i,whisker in enumerate(bp['whiskers']):
        whisker.set(color = colorwhisk[i])
    for i, cap in enumerate(bp['caps']):
        cap.set(color = colorcaps[i])
    for median in bp['medians']:
        median.set(color='grey', linewidth=.8)
    xticks(arange(1,len(labels)+1,1), labels, rotation=rotation, fontsize=10)
    tight_layout()
    return ax

def define_colormap(colors, N):
    cm = LinearSegmentedColormap.from_list('new_cm', colors, N)
    return cm

def format_axes(ax, nfloats_y = '2', nfloats_x = '1'):
    fmt = matplotlib.ticker.StrMethodFormatter("{x}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.'+nfloats_y+'f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.' + nfloats_x + 'f'))

def plot_data_points(x, data, color='k', jitter=0, markersize=2):
    # data is n_observations x n_variables
    n_observations = data.shape[0]
    n_variables = data.shape[1]
    for i in range(n_observations):
        plot(x+random.normal(0,jitter,len(x)), data[i], 'o', color=color, markersize=markersize)
