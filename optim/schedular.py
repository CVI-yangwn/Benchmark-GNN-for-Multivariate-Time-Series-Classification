import torch.optim.lr_scheduler as scheduler

schedulers = {}
def register_sched(opt):
    schedulers[opt.__name__] = opt
    return opt

@register_sched
def CosineAnnealingWarmRestarts(*args, **kargs):
    return scheduler.CosineAnnealingWarmRestarts(*args, **kargs)

@register_sched
def ReduceLROnPlateau(*args, **kargs):
    return scheduler.ReduceLROnPlateau(*args, **kargs)

if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim
    model = nn.Linear(32, 16)
    opt = torch.optim.SGD(model.parameters(), 0.01, 0.9, weight_decay=1e-2)
    sch = schedulers["CosineAnnealingWarmRestarts"](opt, 5, 2)
    print(f"{type(sch).__name__}:\n{sch.state_dict()}")