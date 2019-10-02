import torch.optim as optim
from .scheduler import CosineWithRestarts
from .adabound import AdaBound


def create_optimizer(params, mode='adam', base_lr=1e-3, t_max=10):
    if mode == 'adam':
        optimizer = optim.Adam(params, base_lr)
    elif mode == 'sgd':
        optimizer = optim.SGD(params, base_lr, momentum=0.9, weight_decay=4e-5)
    elif mode == 'adabound':
        optimizer = AdaBound(params, lr=base_lr, final_lr=base_lr*100, gamma=1e-3, eps=1e-8, weight_decay=4e-5)
    else:
        raise NotImplementedError(mode)

    scheduler = CosineWithRestarts(optimizer, t_max)
    #scheduler = None

    return optimizer, scheduler
