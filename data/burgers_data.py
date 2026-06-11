import torch 
def load_data(N_ic, N_bc, N_pde):
    x_ic = torch.FloatTensor(N_ic, 1).uniform_(-1, 1)
    t_ic = torch.zeros(N_ic, 1)

    t_bc = torch.FloatTensor(N_bc, 1).uniform_(0, 1)
    x_bc = torch.cat([-torch.ones(N_bc//2, 1),
                       torch.ones(N_bc//2, 1)])

    x_pde = torch.FloatTensor(N_pde, 1).uniform_(-1, 1)
    t_pde = torch.FloatTensor(N_pde, 1).uniform_(0, 1)
    return x_ic, t_ic, x_bc, t_bc, x_pde, t_pde