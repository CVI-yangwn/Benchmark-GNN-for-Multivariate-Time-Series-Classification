import torch.optim

optimizers = {}
def register_optim(opt):
    optimizers[opt.__name__] = opt
    return opt

@register_optim
def SGD(*args, **kargs):
    return torch.optim.SGD(*args, **kargs)

@register_optim
def Adam(*args, **kargs):
    return torch.optim.Adam(*args, **kargs)

if __name__ == "__main__":
    import torch.nn as nn
    model = nn.Linear(32, 16)
    # opt = SGD(model.parameters(), 0.01, 0.9, weight_decay=1e-2)
    opt = optimizers["SGD"](model.parameters(), 0.01, 0.9, weight_decay=1e-2)
    print(opt)