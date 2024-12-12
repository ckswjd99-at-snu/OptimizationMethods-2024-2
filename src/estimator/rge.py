import torch

@torch.no_grad()
def gradient_estimate_randvec(torch_model, input, target, criterion, num_query=1, eps=1e-3):
    param_dict = dict(torch_model.named_parameters())

    estimated_grad_dict = {pname: torch.zeros_like(param_dict[pname]) for pname in param_dict}

    for i in range(num_query):
        perturb_dict = {pname: torch.randn_like(param_dict[pname]) for pname in param_dict}

        for pname in param_dict:
            param_dict[pname].data.add_(eps, perturb_dict[pname])
        
        loss_pos = criterion(torch_model(input), target)

        for pname in param_dict:
            param_dict[pname].data.sub_(2*eps, perturb_dict[pname])

        loss_neg = criterion(torch_model(input), target)

        for pname in param_dict:
            param_dict[pname].data.add_(eps, perturb_dict[pname])

        estimated_grad_dict = {pname: estimated_grad_dict[pname] + (loss_pos - loss_neg) / (2*eps) * perturb_dict[pname] / num_query for pname in param_dict}

    return estimated_grad_dict

    