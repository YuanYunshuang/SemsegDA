from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from schedulers.schedulers import WarmUpLR, ConstantLR, PolynomialLR

def get_scheduler(optimizer, scheduler_dict):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        scheduler_dict     -- stores name and correspondent parameters for the scheduler
                              possible scheduler: linear | step | plateau | cosine
    Return: a scheduler
    """
    name = scheduler_dict["name"]
    if name == 'step':
        scheduler = StepLR(optimizer, step_size=scheduler_dict["decay_iters"], gamma=0.1)
    elif name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=scheduler_dict['mode'],
                                      factor=scheduler_dict['factor'],
                                      threshold=scheduler_dict['threshold'],
                                      patience=scheduler_dict['patience'])
    elif name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=scheduler_dict["n_iters"], eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

