import torch 
from losses.standard import StandardLoss
def rar_resample(
    model,
    equation,
    x_pde: torch.Tensor,
    t_pde: torch.Tensor,
    k: int,
    n_candidates: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x_candidates = torch.FloatTensor(n_candidates, 1).uniform_(-1, 1)
    t_candidates = torch.FloatTensor(n_candidates, 1).uniform_(0,  1)
    StandardLoss.compute(equation,ic,bc,model,batch: dict)

    pass