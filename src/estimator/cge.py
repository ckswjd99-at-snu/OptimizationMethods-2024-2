import torch

@torch.no_grad()
def gradient_estimate_coord(torch_model, input, target, criterion, num_query=-1, eps=1e-3):
    num_params = sum(p.numel() for p in torch_model.parameters())

    param_dict = dict(torch_model.named_parameters())
    estimated_grad_dict = {pname: torch.zeros_like(param_dict[pname]) for pname in param_dict}
    
    # calculate full gradient
    for pname in param_dict:
        param_size = param_dict[pname].numel()
        for i in range(param_size):
            param_dict[pname].view(-1)[i].add_(eps)
            loss_pos = criterion(torch_model(input), target)
            param_dict[pname].view(-1)[i].sub_(2*eps)
            loss_neg = criterion(torch_model(input), target)
            param_dict[pname].view(-1)[i].add_(eps)
            estimated_grad_dict[pname].view(-1)[i] = (loss_pos - loss_neg) / (2*eps)
    
    # partial gradient
    if num_query != -1 and num_query < num_params:
        flat_grad = torch.cat([p.view(-1) for p in estimated_grad_dict.values()])
        
        chosen_indices = torch.randperm(num_params)[:num_query]
        
        mask = torch.zeros(num_params)
        mask[chosen_indices] = 1
        mask = mask.bool()
        flat_grad = flat_grad * mask
        
        start_idx = 0
        for pname in param_dict:
            param_size = param_dict[pname].numel()
            end_idx = start_idx + param_size
            estimated_grad_dict[pname] = flat_grad[start_idx:end_idx].view(param_dict[pname].shape)
            start_idx = end_idx

    return estimated_grad_dict
