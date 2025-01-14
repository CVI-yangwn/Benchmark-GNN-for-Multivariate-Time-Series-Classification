import sys
sys.path.append("/NAS/yangwennuo/code/MTSC/MTSC-Graph-benchmarking")
from data.config import datasets
import numpy as np
import torch

# ('ArticularyWordRecognition', 200) ('BasicMotions', 10) ('UWaveGestureLibrary', 100) ('Epilepsy', 16)
dataset = ('UWaveGestureLibrary', 100)

dataset_name = dataset[0]
fs = dataset[1]

Dataset = datasets["UEADataset"]
testset = Dataset(name=dataset_name,
                 path="./MTSC-Graph-benchmarking/dataset/Multivariate2018_npz",
                 train=False,
                 logger=None,
                 adjName="complete",
                 nodeName="power_spectral_density",
                 fs=fs,
                 bands=None)
print("load DONE")

def create_grid_subplot(num_plots):
    # 计算子图网格的行数和列数
    # 尽量使行数和列数接近，以形成一个接近正方形的网格

    if num_plots <= 3:
        cols, rows = num_plots, 1
        fig, axs = plt.subplots(rows, cols, figsize=(5*num_plots, 5))
    
    # 如果只有一个子图，将其转换为一个一维数组
    elif num_plots == 1:
        axs = np.array([axs])

    else:
        cols = int(np.ceil(np.sqrt(num_plots)))
        rows = int(np.ceil(num_plots / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(15, 10))

        # 如果子图数量小于网格数量，则隐藏多余的子图
        for i in range(num_plots, rows * cols):
            fig.delaxes(axs[i // cols, i % cols])

    return fig, axs, rows, cols

def draw_distribution(x, dimension, label):
    """
    args:
        x: data
        dimension: the number of subfig
        label: save name
    """
    if isinstance(x, torch.Tensor):
        data_np = x.numpy()
    else:
        data_np = x

    fig, axs, r, c = create_grid_subplot(dimension)
    # fig.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust the space between subplots
    if r == 1:
        for i in range(dimension):
            axs[i].plot(data_np[i], label=f'Dim-{i+1}', linewidth=2)
            axs[i].set_title(f'Dim-{i+1}', fontsize=18, fontweight='bold')
            axs[i].tick_params(axis='x', labelsize=18)
            axs[i].tick_params(axis='y', labelsize=18)
            for a in ['top', 'right', 'left', 'bottom']:
                # axs.spines[a].set_visible(True)
                # axs.spines[a].set_color('black')
                axs[i].spines[a].set_linewidth(1.5)

    else:
        for i in range(dimension):
            row = i // c  # Determine the row index
            col = i % c   # Determine the column index
            axs[row, col].plot(data_np[i], label=f'Dim-{i+1}', linewidth=2)
            # axs[row, col].set_xlabel('X-axis')
            # axs[row, col].set_ylabel('Y-axis')
            axs[row, col].set_title(f'Dim-{i+1}', fontsize=16, fontweight='bold')
            axs[row, col].tick_params(axis='x', labelsize=16)
            axs[row, col].tick_params(axis='y', labelsize=16)
            for a in ['top', 'right', 'left', 'bottom']:
                # axs.spines[a].set_visible(True)
                # axs.spines[a].set_color('black')
                axs[row, col].spines[a].set_linewidth(1.5)

        # axs[row, col].legend()
    plt.suptitle(label, y=0.05)  # Add a title for the whole figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    plt.savefig(f"./pic/distribution/{label}.png", dpi=600)
    plt.cla()

data_dic = {}
dimension = testset[0][-1].size()[0]

import matplotlib.pyplot as plt
for index in range(len(testset)):
    _, x, y, adj, node = testset[10]
    # y = "ori"  # str(y)
    # draw_distribution(x, dimension, f"{dataset_name}-ori-{index}")
    # draw_distribution(node, dimension, f"{dataset_name}-psd-{index}")
    draw_distribution(x, dimension, f"{dataset_name}-ORI")
    draw_distribution(node, dimension, f"{dataset_name}-PSD")
    print("exit for temp")
    exit()
    if y not in data_dic:
        data_dic[y] = []
    data_dic[y].append(node)  # list(tensor(9,72))

r_store = []
for key, data in data_dic.items():
    sums = torch.zeros(size=(dimension,))
    cases = len(data)
    ## --------------------------  cal in cls
    for j in range(len(data)):  # circulate tensors in one class
        d_c = data.copy()
        x_j: torch.Tensor = data[j]
        d_c.pop(j)
        s = torch.zeros(x_j.shape)
        for x_z in d_c:  # get the tensor(9,72) except x_j
            s += torch.abs(x_j - x_z)
        s = torch.sum(s, dim=-1)
        sums += s/cases  # tensor(9,)
        
    r_store.append(sums)  # list[tensor(dimensions,),]

concatenated_tensor = torch.concat(r_store, dim=-1).reshape(-1, dimension)
# print(concatenated_tensor)
draw_distribution(concatenated_tensor, len(concatenated_tensor), f"{dataset_name}-cls-distance")