import torch


def gradient_firstorder(torch_model, input, target, criterion):
    param_dict = dict(torch_model.named_parameters())
    grad_dict = {pname: torch.zeros_like(param_dict[pname]) for pname in param_dict}

    loss = criterion(torch_model(input), target)
    loss.backward()

    for pname in param_dict:
        grad_dict[pname] = param_dict[pname].grad

    return grad_dict