from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("/NAS/yangwennuo/code/MTSC/MTSC-Graph-benchmarking")
from data.config import datasets
import numpy as np
import torch
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

mode = "DE"  # PSD, DE
mode2name = {
    "PSD": "power_spectral_density",
    "DE": "differential_entropy"
}

def fashion_scatter(x_tsne, x_pca, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(10, 5))
    plt.subplot(121, aspect='equal')
    plt.scatter(x_tsne[:,0], x_tsne[:,1], label="t-SNE", c=palette[colors.astype(np.int_)])
    plt.axis('off')
    plt.axis('tight')
    plt.legend()

    plt.subplot(122, aspect='equal')
    plt.scatter(x_pca[:,0], x_pca[:,1], label="PCA", c=palette[colors.astype(np.int_)])
    plt.axis('off')
    plt.axis('tight')
    plt.legend()
    return f

def draw_kl(data, perplexity, y=None, num_cls=None):
    X_tsne = TSNE(n_components=2, random_state=33, perplexity=perplexity).fit_transform(data)
    X_pca = PCA(n_components=2).fit_transform(data)

    f = fashion_scatter(X_tsne, X_pca, y)
    return f

def draw_com(data_ori, data_psd, y=None, num_cls=None, **kwargs):
    ori_tsne = TSNE(n_components=2, random_state=33, perplexity=10).fit_transform(data_ori)
    psd_tsne = TSNE(n_components=2, random_state=33, perplexity=10).fit_transform(data_psd)

    f = scatter(ori_tsne, psd_tsne, y, dataset_name=kwargs.get('dataset_name', 'none'))
    return f

def scatter(ori_tsne, psd_tsne, colors, dataset_name='none'):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(10, 5), facecolor='white')
    plt.suptitle(dataset_name, y = 0.08)
    plt.subplots_adjust(top=0.85)

    ax = plt.subplot(121, aspect='equal', facecolor='white')
    for a in ['top', 'right', 'left', 'bottom']:
        ax.spines[a].set_visible(True)
        ax.spines[a].set_color('black')
        ax.spines[a].set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.scatter(ori_tsne[:,0], ori_tsne[:,1], label="ORI", c=palette[colors.astype(np.int_)])
    plt.grid(False)
    plt.axis('on')
    plt.axis('tight')
    plt.title('ORI')

    ax = plt.subplot(122, aspect='equal', facecolor='white')
    for a in ['top', 'right', 'left', 'bottom']:
        ax.spines[a].set_visible(True)
        ax.spines[a].set_color('black')
        ax.spines[a].set_linewidth(2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.scatter(psd_tsne[:,0], psd_tsne[:,1], label=mode, c=palette[colors.astype(np.int_)])
    plt.grid(False)
    plt.axis('on')
    plt.axis('tight')
    plt.title(mode)

    # plt.legend()
    return f

def draw_tsne4com(dataset_name, fs):
    X = []
    Y = []
    NODE = []
    Dataset = datasets["UEADataset"]
    testset = Dataset(name=dataset_name,
                    path="./MTSC-Graph-benchmarking/dataset/Multivariate2018_npz",
                    train=False,
                    logger=None,
                    adjName="complete",
                    nodeName=mode2name[mode],
                    fs=fs,
                    bands=5)
    for index in range(len(testset)):
        _, x, y, adj, node = testset[index]
        Y.append(y)
        X.append(x.flatten())
        NODE.append(node.flatten())
    Y = np.array(Y)
    X = np.vstack(X)
    NODE = np.vstack(NODE)
    print(dataset_name, X.shape)
    # figure = draw_kl(X, perplexity=30, y=Y)
    # figure.suptitle(f"{dataset_name}_ori")
    # figure.savefig(f'pic/{dataset_name}_ori_tsne.png', dpi=120)

    # figure = draw_kl(NODE, perplexity=30, y=Y)
    # figure.suptitle(f"{dataset_name}_{mode}")
    # figure.savefig(f'pic/{dataset_name}_{mode}_tsne.png', dpi=120)

    figure = draw_com(X, NODE, y=Y, dataset_name=dataset_name)
    figure.savefig(f'pic/tsne/{dataset_name}-{mode}.png', dpi=600)

if __name__ == "__main__":
    # conf = [('ArticularyWordRecognition', 25, 200, 144), ('AtrialFibrillation', 3, 128, 640),
    #         ('BasicMotions', 4, 10, 100), ('Cricket', 12, 184, 1197),
    #         ('DuckDuckGeese', 5, 0, 270), ('EigenWorms', 5, 0, 17984),
    #         ('Epilepsy', 4, 16, 206), ('EthanolConcentration', 4, 0, 1751), 
    #         ('ERing', 6, 0, 65), ('FaceDetection', 2, 250, 62), ('FingerMovements', 2, 100, 50), 
    #         ('HandMovementDirection', 4, 0, 400), ('Handwriting', 26, 0, 152), ('Heartbeat', 2, 0, 405), 
    #         ('Libras', 15, 0, 45), ('LSST', 14, 0, 36), ('MotorImagery', 2, 1000, 3000), 
    #         ('NATOPS', 6, 0, 51), ('PenDigits', 10, 0, 8), ('PEMS-SF', 7, 0, 144), 
    #         ('PhonemeSpectra', 39, 0, 217), ('RacketSports', 4, 10, 30), 
    #         ('SelfRegulationSCP1', 2, 256, 896), ('SelfRegulationSCP2', 2, 256, 1152), 
    #         ('StandWalkJump', 3, 500, 2500), ('UWaveGestureLibrary', 8, 100, 315)]
    conf = [('Epilepsy', 4, 16, 206)]
    for dataset, ncls, fs, length in conf:
        if fs != 0:
            draw_tsne4com(dataset_name=dataset, fs=fs)
    

