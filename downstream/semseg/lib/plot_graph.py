# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.pyplot import *
from PIL import Image

colors = [ 'xkcd:blue',
           'xkcd:red',
           'xkcd:purple',
           'xkcd:orchid',
           'xkcd:orange',
           'xkcd:grey',
           'xkcd:teal',
           'xkcd:sienna',
           'xkcd:azure',
           'xkcd:green',
           'xkcd:black',
           'xkcd:goldenrod']

def bar_plot_insseg(image_name='bar_insseg.png'):

    labels = ['20 Points', '50 Points', '100 Points', '200 points']
    RAND = [14.6, 21.6, 34.0, 43.5]
    Kmeans = [15.6, 24.3, 35.7, 42.3]
    OURS_I = [26.3, 32.6, 39.9, 48.9]
    OURS_S = [25.8, 32.5, 44.2, 48.3]
    OURS_IS = [27.2, 35.7, 43.6, 50.4]

    x = np.array([0,2,4,6])  # the label locations
    width = 1.7  # the width of the bars

    font = {'family' : 'Times New Roman',
            'size'  : 11}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(5.5, 4.5)

    rects1 = ax.bar(x - width*2/5, RAND,    width/5, label='RAND')
    rects2 = ax.bar(x - width*1/5, Kmeans,  width/5, label='Kmeans')
    rects3 = ax.bar(x            , OURS_I,  width/5, label='OURS_I')
    rects4 = ax.bar(x + width*1/5, OURS_S,  width/5, label='OURS_S')
    rects5 = ax.bar(x + width*2/5, OURS_IS, width/5, label='OURS_IS')

    #rects1 = ax.bar(x - width*2/4, points20, width/4, label='20')
    #rects2 = ax.bar(x - width/4, points50, width/4, label='50')
    #rects3 = ax.bar(x + width/4, points100, width/4, label='100')
    #rects4 = ax.bar(x + width*2/4, points200, width/4, label='200')

    ax.plot(np.arange(len(labels)+15)-2, [56.9]*(len(x)+15), '--', linewidth=2.25, color=colors[-1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mAP@0.5')
    ax.set_xlabel('Number of Annotated Points Per Scene')
    ax.set(xlim=[-1, 7], ylim=[0, 61])
    #ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(1.5, 58, '150,000 Annotated Points Per Scene', fontsize=8)

    ax.legend(loc=2)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",fontsize=6,
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    fig.tight_layout()
    plt.show()

    fig.savefig(image_name, dpi=600)
    image = Image.open(image_name)
    image.save(image_name)
 

def bar_plot_sem(image_name='bar_semseg.png'):

    #labels = ['RAND', 'KMEANS', 'OURS_I', 'OURS_S', 'OURS_IS']
    #points20 = [41.9, 45.9, 53.6, 55.5, 53.8]
    #points50 = [53.9, 55.4, 60.7, 60.5, 62.9]
    #points100 = [62.2, 60.6, 65.7, 65.9, 66.9]
    #points200 = [65.5, 64.3, 68.2, 68.2, 69.0]

    labels = ['20 Points', '50 Points', '100 Points', '200 points']
    RAND = [41.9, 53.9, 62.2, 65.5]
    Kmeans = [45.9, 55.4, 60.6, 64.3]
    OURS_I = [55.5, 60.5, 65.9, 68.2]
    OURS_S = [53.6, 60.7, 65.7, 68.2]
    OURS_IS = [53.8, 62.9, 66.9, 69.0]

    x = np.array([0,2,4,6])  # the label locations
    width = 1.7  # the width of the bars

    font = {'family' : 'Times New Roman',
            'size'  : 11}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(5.5, 4.5)

    rects1 = ax.bar(x - width*2/5, RAND,    width/5, label='RAND')
    rects2 = ax.bar(x - width*1/5, Kmeans,  width/5, label='Kmeans')
    rects3 = ax.bar(x            , OURS_I,  width/5, label='OURS_I')
    rects4 = ax.bar(x + width*1/5, OURS_S,  width/5, label='OURS_S')
    rects5 = ax.bar(x + width*2/5, OURS_IS, width/5, label='OURS_IS')

    #rects1 = ax.bar(x - width*2/4, points20, width/4, label='20')
    #rects2 = ax.bar(x - width/4, points50, width/4, label='50')
    #rects3 = ax.bar(x + width/4, points100, width/4, label='100')
    #rects4 = ax.bar(x + width*2/4, points200, width/4, label='200')

    ax.plot(np.arange(len(labels)+15)-2, [72.2]*(len(x)+15), '--', linewidth=2.25, color=colors[-1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('mIoU')
    ax.set_xlabel('Number of Annotated Points Per Scene')
    ax.set(xlim=[-1, 7], ylim=[40, 75])
    #ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(1.5, 73, '150,000 Annotated Points Per Scene', fontsize=8)

    ax.legend()


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",fontsize=6,
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    fig.tight_layout()
    plt.show()

    fig.savefig(image_name, dpi=600)
    image = Image.open(image_name)
    image.save(image_name)
 

def plot_curves(curves, 
                xlabel='% Dataset Labeled\n(ScanNet-5-Recon)',
                xlim=[4, 36],
                xticks=np.arange(5, 35, 5),
                xticklabels=None,
                ylabel='mIoU', 
                ylim=[0.2, 0.65],
                yticks=np.arange(0.2, 0.65, 0.05),
                if_grid=True,
                image_name='test.png'):
    font = {'family' : 'Times New Roman',
            'size'   : 11}
    matplotlib.rc('font', **font)

    fig, subplot = plt.subplots(1,1)
    fig.set_size_inches(8.0, 4.0)
    subplot.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    subplot.set(xticks=xticks, yticks=yticks)
    if xticklabels:
        subplot.axes.set_xticklabels(xticklabels)
    subplot.grid(if_grid)

    for idx, curve in enumerate(curves):
        name = ''
        fmt=''
        marker = ''
        markersize = 10
        linewidth=4.0
        color = colors[idx%len(colors)]


        if 'name' in curve:
            name = curve['name']
        if 'marker' in curve:
            marker = curve['marker']
        if 'markersize' in curve:
            marker_size = curve['markersize']
        if 'color' in curve:
            color = curve['color']
        if 'linewidth' in curve:
            linewidth = curve['linewidth']
        if 'fmt' in curve:
            fmt = curve['fmt']

        x = curve['x']
        y = curve['y']
        
        subplot.plot(x, y, fmt, label=name, marker=marker, markersize=markersize, linewidth=linewidth, color=color)

    subplot.legend(loc='best')
    fig.tight_layout()
    plt.show()
    fig.savefig(image_name, dpi=600)
    image = Image.open(image_name)
    w, h = image.size
    image.crop((75, 75, w - 75, h - 60)).save(image_name)

def shape_contexts_ablation():
    '''
      Variants  & 1024 & 2048 & 4096 \\
        \hline
        1 & 59.7 & 60.7 & 60.1 \\
        2 & 61.4 & 61.6 & 61.9 \\
        4 & 61.7 & 61.8 & 63.0 \\
        8 & 61.2 & 62.1 & 63.4 \\
    '''
    data = [
        {'name': 'No scene contexts', 'x': [1, 2, 3, 4], 'y': [60.5, 60.7, 60.1, 60.6], 'marker': 'o'},
        {'name': '2 Partitions', 'x': [2, 4, 6, 8], 'y': [61.4, 61.6, 61.9, 61.9], 'marker': 'o'},
        {'name': '4 Partitions', 'x': [2, 4, 6, 8], 'y': [61.7, 61.8, 63.0, 62.9], 'marker': '^'},
        {'name': '8 Partitions', 'x': [2, 4, 6, 8], 'y': [61.2, 62.1, 63.4, 63.5], 'marker': 's'},
        {'name': '16 Partitions', 'x': [2, 4, 6], 'y': [61.1, 61.9, 62.6], 'marker': 'p'},
        {'name': '32 Partitions', 'x': [2, 4, 6], 'y': [60.9, 61.7, 62.1], 'marker': '*'},
    ]
    plot_curves(curves=data,
                xlabel='Number of Points',
                ylabel='mAP@0.5',
                xlim=[1.9,8.1],
                xticks=[2,4,6,8],
                xticklabels=[1024, 2048, 4096, 8192],
                ylim=[60.9, 63.5],
                yticks=[60.9, 61.5, 62.5, 63.5],
                if_grid=True, 
                image_name='shape_context_ablation.jpg')



def bar_plot_active(image_name='bar_active.png'):

    labels = ['20 Points', '50 Points', '100 Points', '200 points']
    #kmeans = [734, 1034, 1386, 1688]
    #act =  [1151, 1726, 2153, 2456]
    #total 2873
    kmeans = [0.255, 0.36, 0.482, 0.588]
    act =  [0.401, 0.601, 0.749, 0.855]

    x = np.array([0,2,4,6])  # the label locations
    width = 1.7  # the width of the bars

    font = {'family' : 'Times New Roman',
            'size'  : 11}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8, 4)

    rects1 = ax.bar(x            , kmeans,  width/2, label='kmeans sampling (xyz+rgb)')
    rects2 = ax.bar(x + width*1/2, act,  width/2, label='act. labeling')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of Distinct Objects')
    ax.set_xlabel('Number of Annotated Points Per Scene')
    ax.set(xlim=[-1, 8], ylim=[0.2, 0.9])
    # manipulate
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:.0%}'.format(x) for x in vals])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(False)

    ax.legend(loc=2)


    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.1%}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",fontsize=10,
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()

    fig.savefig(image_name, dpi=600)
    image = Image.open(image_name)
    image.save(image_name)


if __name__=='__main__':
    #shape_contexts_ablation()
    #bar_plot_insseg()
    #bar_plot_semseg()
    #bar_plot_active()

