import sys
sys.path.append("/data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking")
from data.config import datasets
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

# ('HandMovementDirection', 4, 0, 400)
dataset = ('HandMovementDirection', 4, 0, 400)
# "mutual_information" "abspearson" "complete" "diffgraphlearn"
dataset_name = dataset[0]
fs = dataset[1]

def draw_heatmap(adj):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.heatmap(adj.cpu().detach().numpy())
    # 不要坐标轴刻度
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 不要边缘太多的空白
    plt.tight_layout()
    # 进一步减少边缘
    plt.savefig(f'/data/yangwennuo/code/MTSC/pic/adj/{dataset_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()


Dataset = datasets["UEADataset"]
testset = Dataset(name=dataset_name,
                path="./MTSC-Graph-benchmarking/dataset/Multivariate2018_npz",
                train=False,
                logger=None,
                adjName="complete", 
                nodeName="raw",
                fs=fs,
                bands=None)
print("load DONE")
model = torch.load(f"/data/yangwennuo/code/MTSC/pic/model/{dataset_name}/195.pt").to("cuda:0")

adjs = []
# for index in range(len(testset)):

_, x, y, adj, node = testset[0]
dim, length = x.shape
# x in this part is node actually
x = node.unsqueeze(0).to("cuda:0")
print(x.shape)
x = model.t_embed(x)
gx = model.gproj(x)  # GlobalProj
e = model.edge_extractor(x, gx)  # [B,V*V,T]
e = e.reshape(dim, dim, -1)

print(e.shape)
# stacked_adjs = torch.cat(adjs, dim=0)
# mean_adj = torch.mean(stacked_adjs, dim=0)

# print(stacked_adjs.shape)

# draw_heatmap(mean_adj)
