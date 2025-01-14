import os
from tqdm import tqdm
from config import get_cfg_defaults, merge_from_file_withoutsafe
from nn.config import models
from data.config import datasets
from optim.optimizer import optimizers
from optim.schedular import schedulers
from data.utils.adjacency import adj_matrix
from data.utils.node import node_feature
import torch
import numpy as np
import logging
import time
import argparse
from typing import Callable
from torch.utils.data import DataLoader
import torch.nn as nn
# from utils import train_adj, test_adj, train_node, test_node
from utils import lambda_train, lambda_test
from train import train
from test import test

lambda_train.reset()
lambda_test.reset()
# train_node.reset()
# test_node.reset()
# train_adj.reset()
# test_adj.reset()

def main(log_path: str, conf_file: str, save_path: str, gpu: int, seed: int) -> Callable:
    
    #==================通用设置====================#

    # print("=====> Loading config...")
    # end = time.time()
    cfg = get_cfg_defaults()
    merge_from_file_withoutsafe(cfg, conf_file)
    cfg.SYSTEM.GPU = gpu

    cfg.freeze()
    # load_config_time = time.time() - end
    # print("Config loading time: {:.4f}s".format(load_config_time))


    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.GPU

    # torch.manual_seed(cfg.SYSTEM.SEED)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = os.path.splitext(os.path.basename(conf_file))[0]
    # save_file = os.path.join(save_path, task_name+".pth")
    # save_file_train = os.path.join(save_path, task_name+"_tr"+".pth")
    # save_file_loss = os.path.join(save_path, task_name+"_loss"+".pth")
    # save_file_trloss = os.path.join(save_path, task_name+"_trloss"+".pth")

    logger = logging.getLogger(task_name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d] %(message)s")
        log_file = log_path
        ##----------- to log repeated exp with random seed
        # log_file = os.path.join(log_path, task_name+".log")
        # if os.path.exists(log_file):
        #     for i in range(1, 100, 1):
        #         log_file = os.path.join(log_path, task_name+"-"+str(i)+".log")
        #         if not os.path.exists(log_file):
        #             break
        #     else:
        #         raise ValueError("logs is too many")
        file_handler = logging.FileHandler(log_file, "w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # logger.addHandler(stream_handler)

    #==================读取数据====================#
    logger.info("=====> Preparing data...")

    try:
        fs = cfg.DATASET.fs
        bands = cfg.DATASET.bands
    except AttributeError:
        fs = 0
        bands = None

    Dataset = datasets[cfg.DATASET.CLASS]
    end = time.time()
    trainset = Dataset(**dict(cfg.DATASET.PARAM), train=True, logger=logger, 
                       adjName=cfg.GRAPH.ADJ_MATRIX, nodeName=cfg.GRAPH.NODE, fs=fs, bands=bands)
    testset = Dataset(**dict(cfg.DATASET.PARAM), train=False, logger=logger, 
                      adjName=cfg.GRAPH.ADJ_MATRIX, nodeName=cfg.GRAPH.NODE,fs=fs, bands=bands)
    # trainloader = DataLoader(trainset, cfg.EXPERIMENT.BATCH_SIZE, True,
    #                         num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    # testloader = DataLoader(testset, cfg.EXPERIMENT.BATCH_SIZE, False,
    #                         num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    trainloader = DataLoader(trainset, cfg.EXPERIMENT.BATCH_SIZE, True,
                            num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, cfg.EXPERIMENT.BATCH_SIZE, False,
                            num_workers=8, pin_memory=True)
    prepare_time = time.time() - end
    logger.info("Data preparing time: {:.4f}s".format(prepare_time))

    #==================定义模型====================#

    logger.info("=====> Loading model...")
    end = time.time()
    model_params = dict(cfg.MODEL.PARAM)
    model_params["num_nodes"] = trainset.num_nodes()
    model = models[cfg.MODEL.CLASS](**model_params).to(device)
    if torch.cuda.device_count() > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        model = nn.DataParallel(model)
    load_model_time = time.time() - end
    logger.info("Model loading time: {:.4f}s".format(load_model_time))

    #==================训练设置====================#

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizers[cfg.EXPERIMENT.OPTIMIZER.CLASS]
    optimizer = optimizer(model.parameters(),
                        **dict(cfg.EXPERIMENT.OPTIMIZER.PARAM))
    scheduler = schedulers[cfg.EXPERIMENT.SCHEDULER.CLASS]
    scheduler = scheduler(optimizer, **dict(cfg.EXPERIMENT.SCHEDULER.PARAM))

    #==================训练过程====================#

    func2path = {
        "complete": "COM",
        "abspearson": "PCC",
        "mutual_information": "MI",
        "phase_locking_value": "PLV",
        "differential_entropy": "DE",
        "power_spectral_density": "PSD"
    }

    datapath = os.path.expanduser(cfg.DATASET.PARAM.path)
    # try:
    #     adj_path = os.path.join(datapath,
    #                             func2path[cfg.GRAPH.ADJ_MATRIX],
    #                             f"{cfg.DATASET.PARAM.name}")
    # except KeyError:
    #     adj_path = None
    # if adj_path is not None:
    #     train_adj.set_path(adj_path)
    #     test_adj.set_path(adj_path)
    # adjacency = adj_matrix[cfg.GRAPH.ADJ_MATRIX]

    # try:
    #     node_path = os.path.join(datapath,
    #                              func2path[cfg.GRAPH.NODE],
    #                              f"{cfg.DATASET.PARAM.name}")
    # except KeyError:
    #     node_path = None
    # if node_path is not None:
    #     train_node.set_path(node_path)
    #     test_node.set_path(node_path)
    # node_feat = node_feature[cfg.GRAPH.NODE]
    
    try:
        lambda_path = os.path.join(datapath,
                                "lambda",
                                func2path[cfg.GRAPH.ADJ_MATRIX],
                                f"{cfg.DATASET.PARAM.name}")
    except KeyError:
        lambda_path = None
    lambda_train.set_path(lambda_path)
    lambda_test.set_path(lambda_path)


    best_accuracy = 0.
    # best_train_accuracy = 0.
    # best_train_loss = float("inf")
    # best_loss = float("inf")
    # best_epoch = 0
    accuracies = []
    train_losses = []
    checkpoints = []
    logger.info(f"Experiment setting:\n"
                f"model:\n{model}\n"
                f"loss_fn:\n{loss_fn}\n"
                f"optimizer:\n{type(optimizer).__name__} "
                f"{optimizer.state_dict()}\n"
                f"scheduler:\n{type(scheduler).__name__} "
                f"{scheduler.state_dict()}\n")
    for epoch in tqdm(range(cfg.EXPERIMENT.EPOCHS)):
        end = time.time()
        logger.info(f"==========Epoch{epoch}==========")
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        logger.info(f"Learning rate: {lr}")
        train_loss, train_accuracy, train_pred = \
        train(trainloader, model, loss_fn, optimizer, scheduler, epoch, logger,
            device)
        loss, accuracy, pred = test(testloader, model, loss_fn, logger, device)
        # if loss != loss:
        #     logger.error(f"Loss is {loss}!")
        #     return -1
        if train_loss != train_loss:
            logger.error(f"Loss is {loss}!")
            return -1
        # 不同的scheduler可能需要传不同的参数
        scheduler.step(train_loss)  # ReduceLROnPlateau
        # scheduler.step()  # CosineAnnealingWarmRestarts

        logger.info(f"Epoch time: {time.time() - end:.4f}\n")

        accuracies.append(accuracy)
        train_losses.append(train_loss)
        
        # checkpoint = {
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "scheduler_state_dict": scheduler.state_dict(),
        #     "prediction": pred
        # }
        # save_file = os.path.join(save_path, task_name + "-" + str(epoch) +".pth")
        # checkpoints.append((checkpoint, save_file))
        # torch.save(checkpoint, save_file)

    # logger.info(f"best_accuracy: {best_accuracy:.2f} at epoch {best_epoch}\n")
    accuracies = np.array(accuracies)
    train_losses = np.array(train_losses)
    es = np.abs((train_losses[1:] - train_losses[:-1]) / (train_losses[:-1] + 1e-5)) <= 0.1
    st = np.full(es.size - 3, False)
    st[:] = es[:-3] & es[1:-2] & es[2:-1] & es[3:]
    accm = np.stack([accuracies[:-4], accuracies[1:-3], accuracies[2:-2], accuracies[3:-1], accuracies[4:]])
    avg = accm.mean(axis=0)
    avg[~st] = -1.0
    std = accm.std(axis=0)
    epoch = len(avg) - np.argmax(avg[::-1]) - 1
    best_epoches = [epoch, epoch+1, epoch+2, epoch+3, epoch+4]
    best_epoch = epoch + np.argmax(accuracies[best_epoches])
    best_accuracy = max(accuracies[best_epoches])
    
    logger.info(f"Best accuray is {best_accuracy} at eopch {best_epoch}")
    logger.info(f"Best avg accuracy is {avg[epoch]}({std[epoch]}) at epoches {best_epoches}")
    logger.info(f"{task_name} done!")


    return best_accuracy

# python main.py --log_path /raid/zhouyz/log/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary --conf_file /raid/zhouyz/config/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary/raw-phase_locking_value-gcn-1-UWaveGestureLibrary.yml --save_path /raid/zhouyz/model/raw-phase_locking_value-gcn-1/raw-phase_locking_value-gcn-1-UWaveGestureLibrary --gpu 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default="/data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking/train.log", help="日志保存位置")
    parser.add_argument("--conf_file", default="/data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking/config/new/ArticularyWordRecognition/gcn-complete-raw.yml", help="配置文件")
    parser.add_argument("--save_path", default="/data/yangwennuo/code/MTSC/MTSC-Graph-benchmarking/models", help="模型保存位置")
    parser.add_argument("--gpu", default="1", help="显卡id")
    parser.add_argument("--seed", default="42", help="随机种子") 
    args = parser.parse_args()
    # args = parser.parse_args()
    log_path = os.path.expanduser(args.log_path)
    conf_file = os.path.expanduser(args.conf_file)
    save_path = os.path.expanduser(args.save_path)
    gpu = args.gpu
    seed = os.path.expanduser(args.seed)

    ## 这步在train.sh里替代了
    # if not os.path.exists(log_path):
    #     os.makedirs(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main(log_path, conf_file, save_path, gpu, seed)
