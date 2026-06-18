import torch 
from equations import burgers_eq 
def rar_resample(
    model,
    equation,
    x_pde: torch.Tensor,
    t_pde: torch.Tensor,
    k: int,
    n_candidates: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x_candidates = torch.FloatTensor(n_candidates, 1).uniform_(-1, 1).to(device)
    t_candidates = torch.FloatTensor(n_candidates, 1).uniform_(0,  1).to(device)
    res_raw = equation.pde_residual_raw(model, x_candidates, t_candidates)
    
    res_raw = torch.abs(res_raw)
    res_raw = res_raw.squeeze()
    idx = torch.topk(res_raw, k=k).indices
    x_new = x_candidates[idx].detach()
    t_new = t_candidates[idx].detach()
    x_pde_new = torch.cat([x_pde.detach(), x_new], dim=0)
    t_pde_new = torch.cat([t_pde.detach(), t_new], dim=0)
    return x_pde_new, t_pde_new


