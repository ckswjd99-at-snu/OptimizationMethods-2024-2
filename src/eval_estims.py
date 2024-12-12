import torch

from estimator import cosine_similarity, gradient_firstorder, gradient_estimate_randvec, gradient_estimate_coord, gradient_estimate_grouped

NUM_REPEAT = 10
PERTURB_SIZE = 1e-4
NUM_QUERY = 20

class TempModel(torch.nn.Module):
    def __init__(self):
        super(TempModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# random dataset
x = torch.randn(10)
y = torch.randn(10)

model = TempModel()
model.train()

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

criterion = torch.nn.MSELoss()

# calculate real gradient
real_grad_dict = gradient_firstorder(model, x, y, criterion)

# calculate estimated gradient
cossim_rge_sum = 0
cossim_cge_sum = 0
cossim_gge_sum = 0

for _ in range(NUM_REPEAT):
    num_query = NUM_QUERY
    eps = PERTURB_SIZE
    # group_sizes = [1] * num_params  # CGE
    # group_sizes = [num_params]      # RGE
    group_sizes = [1, 1, 1, 217]           # GGE
    alpha = 0.95

    estimated_grad_rge = gradient_estimate_randvec(model, x, y, criterion, num_query, eps)
    estimated_grad_cge = gradient_estimate_coord(model, x, y, criterion, num_query, eps)
    estimated_grad_gge = gradient_estimate_grouped(model, x, y, criterion, group_sizes, num_query // len(group_sizes), eps, alpha, ref=real_grad_dict)

    cos_sim_rge = cosine_similarity(real_grad_dict, estimated_grad_rge)
    cos_sim_cge = cosine_similarity(real_grad_dict, estimated_grad_cge)
    cos_sim_gge = cosine_similarity(real_grad_dict, estimated_grad_gge)

    cossim_rge_sum += cos_sim_rge
    cossim_cge_sum += cos_sim_cge
    cossim_gge_sum += cos_sim_gge

# print cosine similarity
print(f"RGE: {cossim_rge_sum / NUM_REPEAT}")
print(f"CGE: {cossim_cge_sum / NUM_REPEAT}")
print(f"GGE: {cossim_gge_sum / NUM_REPEAT}")
