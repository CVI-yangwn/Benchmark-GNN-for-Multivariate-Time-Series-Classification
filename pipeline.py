from config import get_cfg_defaults
from nn.config import models
from data.config import datasets
from optim.optimizer import optimizers
from optim.schedular import schedulers
from data.utils.adjacency import adj_matrix
from data.utils.node import node_feature
import torch
import os
import logging
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import train, test

def pipeline(conf_file: str, log_path: str = ".") -> None:
    #==================通用设置====================#

    print("=====> Loading config...")
    end = time.time()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(conf_file)
    cfg.freeze()
    load_config_time = time.time() - end
    print("Config loading time: {:.4f}s".format(load_config_time))


    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.SYSTEM.GPU)
    torch.manual_seed(cfg.SYSTEM.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = os.path.splitext(os.path.basename(conf_file))[0]
    save_file = os.path.join(cfg.EXPERIMENT.SAVE_PATH, task_name+"pth")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    log_file = os.path.join(log_path, task_name+".log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    #==================读取数据====================#

    Dataset = datasets[cfg.DATASET.CLASS]
    logger.info("=====> Preparing data...")
    end = time.time()
    trainset = Dataset(**dict(cfg.DATASET.PARAM), train=True)
    testset = Dataset(**dict(cfg.DATASET.PARAM), train=False)
    trainloader = DataLoader(trainset, cfg.EXPERIMENT.BATCH_SIZE, True,
                            num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    testloader = DataLoader(testset, len(testset), False,
                            num_workers=cfg.SYSTEM.NUM_WORKERS, pin_memory=True)
    prepare_time = time.time() - end
    logger.info("Data preparing time: {:.4f}s".format(prepare_time))

    #==================定义模型====================#

    logger.info("=====> Loading model...")
    end = time.time()
    model = models[cfg.MODEL.CLASS](**dict(cfg.MODEL.PARAM)).to(device)
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

    adjacency = adj_matrix[cfg.GRAPH.ADJ_MATRIX]
    node_feat = node_feature[cfg.GRAPH.NODE]
    best_accuracy = 0.
    best_epoch = 0
    logger.info(f"Experement settting:\n"
                f"model:\n{model}\n"
                f"loss_fn:\n{loss_fn}\n"
                f"optimizer:\n{type(optimizer).__name__} "
                f"{optimizer.state_dict()}\n"
                f"scheduler:\n{type(scheduler).__name__} "
                f"{scheduler.state_dict()}\n")
    for epoch in range(cfg.EXPERIMENT.EPOCHS):
        end = time.time()
        print(f"==========Epoch{epoch}==========")
        train(trainloader, model, loss_fn, optimizer, scheduler, epoch, logger,
            device, adjacency, node_feat)
        _, accuracy = test(testloader, model, loss_fn, logger, device,
                            adjacency, node_feat)
        logger.info(f"Epoch time: {time.time() - end:.4f}\n")
        
        if accuracy > best_accuracy:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, save_file)
            best_accuracy = accuracy
            best_epoch = epoch
    logger.info(f"best_accuracy: {best_accuracy:.2f} at epoch {best_epoch}\n")

if __name__ == "__main__":
    conf_file = "config/gcn.yml"
    pipeline(conf_file, ".")