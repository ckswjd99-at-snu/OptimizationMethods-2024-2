import torch
from .utils import cosine_similarity

def calc_group_dict(grad_dict, group_sizes):
    num_params = sum(p.numel() for p in grad_dict.values())
    
    grad_abs_flat = torch.cat([p.view(-1) for p in grad_dict.values()]).abs()
    grad_abs_flat_sorted = grad_abs_flat.clone().sort(descending=True)[0]
    thres_indices = [sum(group_sizes[:i])-1 for i in range(len(group_sizes)+1)]
    thres_indices = thres_indices[1:]
    thres_values = grad_abs_flat_sorted[thres_indices]

    group_flat = torch.zeros(num_params)
    for i in range(len(group_sizes)-1, -1, -1):
        group_flat[grad_abs_flat >= thres_values[i]] = i

    group_dict = {}
    start_idx = 0
    for pname, p in grad_dict.items():
        param_size = p.numel()
        end_idx = start_idx + param_size
        group_dict[pname] = group_flat[start_idx:end_idx].view(p.shape).long()
        start_idx = end_idx

    return group_dict

def rand_group_dict(grad_dict, group_sizes):
    rand_momentum_dict = {pname: torch.randn_like(grad_dict[pname]) for pname in grad_dict}
    group_dict = calc_group_dict(rand_momentum_dict, group_sizes)
    return group_dict


@torch.no_grad()
def gradient_estimate_grouped(torch_model, input, target, criterion, group_sizes, num_iter=1, eps=1e-3, alpha=0.9, ref=None):
    num_params = sum(p.numel() for p in torch_model.parameters())
    
    assert sum(group_sizes) == num_params, "group_sizes must sum up to the number of parameters"

    param_dict = dict(torch_model.named_parameters())
    estimated_grad_dict = {pname: torch.zeros_like(param_dict[pname]) for pname in param_dict}
    group_dict = rand_group_dict(estimated_grad_dict, group_sizes)
    group_flat = torch.cat([gd.view(-1) for gd in group_dict.values()])
    igrad_flat = torch.cat([gd.view(-1) for gd in estimated_grad_dict.values()])
    # print(f"initial")
    # print(f"  group: {group_flat}")
    # print(f"  grad: {igrad_flat}")

    for i in range(num_iter):
        iter_grad_dict = {pname: torch.zeros_like(param_dict[pname]) for pname in param_dict}
        perturb_dict = {pname: torch.randn_like(param_dict[pname]) for pname in param_dict}

        for group_id in range(len(group_sizes)):
            perturb_norm = 0
            for pname in param_dict:
                perturb_mask = (group_dict[pname] == group_id).float()
                perturb_norm += (perturb_dict[pname] * perturb_mask).norm().item() ** 2
            perturb_norm = perturb_norm ** 0.5

            for pname in param_dict:
                perturb_mask = (group_dict[pname] == group_id).float()
                perturb_noise = perturb_dict[pname] * perturb_mask / perturb_norm
                param_dict[pname].data.add_(eps, perturb_noise)

            loss_pos = criterion(torch_model(input), target)

            for pname in param_dict:
                perturb_mask = (group_dict[pname] == group_id).float()
                perturb_noise = perturb_dict[pname] * perturb_mask / perturb_norm
                param_dict[pname].data.sub_(2*eps, perturb_noise)

            loss_neg = criterion(torch_model(input), target)

            for pname in param_dict:
                perturb_mask = (group_dict[pname] == group_id).float()
                perturb_noise = perturb_dict[pname] * perturb_mask / perturb_norm
                param_dict[pname].data.add_(eps, perturb_noise)

            for pname in param_dict:
                perturb_mask = (group_dict[pname] == group_id).float()
                iter_grad_dict[pname] += (loss_pos - loss_neg) / (2*eps) * perturb_dict[pname] * perturb_mask / perturb_norm

        for pname in param_dict:
            # if i == 0:
            #     estimated_grad_dict[pname] = iter_grad_dict[pname]
            # else:
            #     estimated_grad_dict[pname] = alpha * estimated_grad_dict[pname] + (1-alpha) * iter_grad_dict[pname]
            # estimated_grad_dict[pname] = alpha * estimated_grad_dict[pname] + (1-alpha) * iter_grad_dict[pname]
            estimated_grad_dict[pname] += iter_grad_dict[pname] / num_iter

        group_flat = torch.cat([gd.view(-1) for gd in group_dict.values()])
        igrad_flat = torch.cat([gd.view(-1) for gd in iter_grad_dict.values()])
        # print(f"iter {i+1}")
        # print(f"  group: {group_flat}")
        # print(f"  grad: {igrad_flat}")
        group_dict = calc_group_dict(estimated_grad_dict, group_sizes)

        if ref is not None:
            cos_sim = cosine_similarity(ref, estimated_grad_dict)
            # print(f"iter {i+1}: {cos_sim}")

    return estimated_grad_dict

    