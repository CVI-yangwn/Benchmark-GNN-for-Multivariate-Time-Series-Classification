import sys
sys.path.append("/data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking")
from data.config import datasets
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

# ('ArticularyWordRecognition', 200) ('BasicMotions', 10) ('UWaveGestureLibrary', 100) ('Epilepsy', 16)
# ('MotorImagery', 2, 1000, 3000) ('RacketSports', 4, 10, 30) ('Heartbeat', 2, 0, 405)
# ('FingerMovements', 2, 100, 50)
# ('Heartbeat', 2, 0, 405) ('PEMS-SF', 7, 0, 144)
# ('FingerMovements', 2, 100, 50) ('NATOPS', 6, 0, 51)
dataset = ('FingerMovements', 2, 100, 50)
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
    plt.savefig(f'/data/yangwennuo/code/MTSC/pic/adj/{dataset_name}_{adj_mode}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.clf()
# "mutual_information","abspearson","complete","diffgraphlearn"
for adj_mode in ["mutual_information","abspearson","complete","diffgraphlearn"]:
    Dataset = datasets["UEADataset"]
    if adj_mode != "diffgraphlearn":
        testset = Dataset(name=dataset_name,
                        path="./MTSC-Graph-benchmarking/dataset/Multivariate2018_npz",
                        train=False,
                        logger=None,
                        adjName=adj_mode, 
                        nodeName="raw",
                        fs=fs,
                        bands=None)
        print("load DONE")

        adjs = []
        for index in range(len(testset)):
            _, x, y, adj, node = testset[index]
            adjs.append(adj.unsqueeze(2))
        stacked_adjs = torch.cat(adjs, dim=2)
        mean_adj = torch.mean(stacked_adjs, dim=2)

    else:
        testset = Dataset(name=dataset_name,
                        path="./MTSC-Graph-benchmarking/dataset/Multivariate2018_npz",
                        train=False,
                        logger=None,
                        adjName="complete", 
                        nodeName="raw",
                        fs=fs,
                        bands=None)
        print("load DONE and infer for diff")
        model = torch.load(f"/data/yangwennuo/code/MTSC/pic/model/{dataset_name}/best.pt").to("cuda:7")
        
        adjs = []
        for index in range(len(testset)):
            _, x, y, adj, node = testset[index]

            adj = model.graphlearn(torch.from_numpy(x).unsqueeze(0).to("cuda:7"))
            adjs.append(adj)
        stacked_adjs = torch.cat(adjs, dim=0)
        mean_adj = torch.mean(stacked_adjs, dim=0)
    
    print(stacked_adjs.shape)
    n = mean_adj.size(0)
    # 生成单位矩阵
    eye_matrix = torch.eye(n, dtype = torch.bool)
    # 将对角线上的值设为0
    mean_adj[eye_matrix] = 0
    if adj_mode=="complete":
        normalized_mean_adj = mean_adj
    else:
        min_val = torch.min(mean_adj)
        max_val = torch.max(mean_adj)
        normalized_mean_adj = (mean_adj - min_val) / (max_val - min_val)
    print(normalized_mean_adj)
    draw_heatmap(mean_adj)
