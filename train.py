import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Callable, Optional, List, Tuple
from logging import Logger
from nn.chebnet import ChebNet
from utils import lambda_train
# from utils import train_adj, train_node

def train(trainloader: DataLoader,
          model: torch.nn.Module,
          loss_fn: Callable,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epoch: int,
          logger: Logger,
          device: torch.device,
        #   adjacency: Callable,
        #   node_feat: Callable,
        #   fs: int = 0,
        #   bands: Optional[List[Tuple[float, float]]] = None
          ) -> Tuple[float, float, List[int]]:
    # if adj_class.path_not_none():
    #     adj_class.set_path(f"{adj_class.path}_TRAIN.pt")
    # train_adj.set_adj(adjacency)
    # train_node.set_node_feat(node_feat)
    model.train()
    train_correct = 0
    total_samples = 0
    end = time.time()
    log_step = int(len(trainloader) / 2)
    for i, (idx, input, label, adj, node) in enumerate(trainloader):
        # if fs != 0 and bands is not None:
        #     # node = torch.stack([node_feat(g, fs, bands) for g in input])
        #     node = train_node.node(input, fs, bands, idx, "_TRAIN.pt")
        # elif fs != 0:
        #     # node = torch.stack([node_feat(g, fs) for g in input])
        #     node = train_node.node(input, fs, None, idx, "_TRAIN.pt")
        # else:
        #     # node = torch.stack([node_feat(g) for g in input])
        #     node = train_node.node(input, None, None, idx, "_TRAIN.pt")
        # adj = torch.stack([adjacency(g) for g in input])
        # adj = train_adj.adj(input)
        # logger.info(adj[0, 0, :10])
        # input = node

        input, label = input.to(device), label.to(device)
        adj = adj.to(device)
        node = node.to(device)
        data_time = time.time() - end
        end = time.time()

        if isinstance(model, ChebNet):
            lambda_max = lambda_train.lambda_max(idx, "_TRAIN.pt")
            output = model(adj, node, input, lambda_max)
        else:
            output = model(adj, node, input)
        # output = output - output.max(dim=1, keepdim=True)[0]
        loss = loss_fn(output, label)
        # assert loss < 1000000, "loss explosion"
        # if epoch == 11:
        #     pass
        optimizer.zero_grad()
        # if not torch.isnan(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000, 2)
        # assert torch.all(model.mlp[4].weight.grad < 100), "grad explosion"
        optimizer.step()
        # scheduler.step()
        train_time = time.time() - end
        # pred = output.argmax(dim=1, keepdim=True)
        # train_correct += pred.eq(label.view_as(pred)).sum().item()
        # train_loss += loss.item() * input.shape[0]
        # total_samples += input.shape[0]

        if (log_step != 0) and (i % log_step == 0):
            logger.info( "Train Epoch: {} [{}/{} ({:.0f}%)] "\
                         "Loss: {:.6f} Data loading time: {:.4f}s "\
                         "Train time: {:.4f}s".format(
                epoch, total_samples, len(trainloader.dataset),
                100. * total_samples / len(trainloader.dataset), loss.item(),
                data_time, train_time))
        
        end = time.time()

    # train_loss = 0
    train_loss = []
    total_samples = 0
    preds = []
    model.eval()
    with torch.no_grad():
        for i, (idx, input, label, adj, node) in enumerate(trainloader):
            # if fs != 0 and bands is not None:
            #     # node = torch.stack([node_feat(g, fs, bands) for g in input])
            #     node = train_node.node(input, fs, bands, idx, "_TRAIN.pt")
            # elif fs != 0:
            #     # node = torch.stack([node_feat(g, fs) for g in input])
            #     node = train_node.node(input, fs, None, idx, "_TRAIN.pt")
            # else:
            #     # node = torch.stack([node_feat(g) for g in input])
            #     node = train_node.node(input, None, None, idx, "_TRAIN.pt")
            # adj = torch.stack([adjacency(g) for g in input])
            # adj = train_adj.adj(input, idx, "_TRAIN.pt")
            # input = node

            input, label = input.to(device), label.to(device)
            adj = adj.to(device)
            node = node.to(device)

            if isinstance(model, ChebNet):
                lambda_max = lambda_train.lambda_max(idx, "_TRAIN.pt")
                output = model(adj, node, input, lambda_max)
            else:
                output = model(adj, node, input)
            loss = loss_fn(output, label)
            pred = output.argmax(dim=1, keepdim=True)
            if (pred.numel() == 1):
                preds.append(pred.item())
            else:
                preds.extend(pred.squeeze().tolist())
            train_correct += pred.eq(label.view_as(pred)).sum().item()
            # train_loss += loss.item() * input.shape[0]
            train_loss += [loss.item()] * input.shape[0]
            total_samples += input.shape[0]

    # train_loss /= total_samples
    train_loss = np.array(train_loss)
    avg_loss = (train_loss / total_samples).sum()
    train_accuracy = 100. * train_correct / total_samples
    infer_time = time.time() - end
    logger.info("Train set: Average loss: {:.4f}, "\
                "Accuracy: {}/{} ({:.2f}%) Inference time: {:.4f}s".format(
        avg_loss, train_correct, total_samples, train_accuracy, infer_time))
    return avg_loss, train_accuracy, preds
