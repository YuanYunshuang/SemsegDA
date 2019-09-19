import  torch

def zero_mean(model, all=True):
    """
    Substract mean for Conv2d layer weights.
    :param model: a model containing Conv2d layer, or a single Conv2d module
    :param all: the default is "True", which means all Conv2d modules in this will be zero-meaned, otherwise the operation
                will only be done for the first Conv2d module
    :return: model with expected zero mean filters
    """
    if all:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)

    else:
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data = m.weight.data - torch.mean(m.weight.data)
                break

    return model
