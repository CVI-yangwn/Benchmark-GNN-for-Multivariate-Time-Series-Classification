import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple, List
from logging import Logger
from nn.chebnet import ChebNet
from utils import lambda_test
# from utils import test_adj, test_node

def test(testloader: DataLoader,
         model: torch.nn.Module,
         loss_fn: Callable,
         logger: Logger,
         device: torch.device,
        #  adjacency: Callable,
        #  node_feat: Callable,
        #  fs: int = 0,
        #  bands: Optional[List[Tuple[float, float]]] = None
         ) -> Tuple[float, float]:
    # test_adj.set_adj(adjacency)
    # test_node.set_node_feat(node_feat)
    model.eval()
    # test_loss = 0
    test_loss = []
    test_correct = 0
    total_samples = 0
    preds = []
    end = time.time()
    with torch.no_grad():
        for idx, input, label, adj, node in testloader:
            # if fs != 0 and bands is not None:
            #     # node = torch.stack([node_feat(g, fs, bands) for g in input])
            #     node = test_node.node(input, fs, bands, idx, "_TEST.pt")
            # elif fs != 0:
            #     # node = torch.stack([node_feat(g, fs) for g in input])
            #     node = test_node.node(input, fs, None, idx, "_TEST.pt")
            # else:
            #     # node = torch.stack([node_feat(g) for g in input])
            #     node = test_node.node(input, None, None, idx, "_TEST.pt")
            # input = input.to(device)
            # adj = torch.stack([adjacency(g) for g in input])
            # adj = test_adj.adj(input, idx, "_TEST.pt")
            # input = node

            input, label = input.to(device), label.to(device)
            adj = adj.to(device)
            node = node.to(device)

            if isinstance(model, ChebNet):
                lambda_max = lambda_test.lambda_max(idx, "_TEST.pt")
                output = model(adj, node, input, lambda_max)
            else:
                output = model(adj, node, input)
            # test_loss += loss_fn(output, label).item() * input.shape[0]
            test_loss += [loss_fn(output, label).item()] * input.shape[0]

            pred = output.argmax(1, keepdim=True)
            if (pred.numel() == 1):
                preds.append(pred.item())
            else:
                preds.extend(pred.squeeze().tolist())
            test_correct += pred.eq(label.view_as(pred)).sum().item()
            total_samples += input.shape[0]

    # test_loss /= total_samples
    test_loss = np.array(test_loss)
    avg_loss = (test_loss / total_samples).sum()
    test_accuracy = 100. * test_correct / total_samples
    infer_time = time.time() - end
    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) "\
                "Inference time: {:.4f}s".format(
        avg_loss, test_correct, total_samples, test_accuracy, infer_time))
    return avg_loss, test_accuracy, preds