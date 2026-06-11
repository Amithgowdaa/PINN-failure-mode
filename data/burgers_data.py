import torch
import scipy.io


def load_training_data(
    N_ic: int,
    N_bc: int,
    N_pde: int
):
    x_ic = torch.FloatTensor(N_ic, 1).uniform_(-1, 1)
    t_ic = torch.zeros(N_ic, 1)

    t_bc = torch.FloatTensor(N_bc, 1).uniform_(0, 1)

    x_bc = torch.cat([
        -torch.ones(N_bc // 2, 1),
         torch.ones(N_bc // 2, 1)
    ])

    x_pde = torch.FloatTensor(N_pde, 1).uniform_(-1, 1)
    t_pde = torch.FloatTensor(N_pde, 1).uniform_(0, 1)

    return (
        x_ic,
        t_ic,
        x_bc,
        t_bc,
        x_pde,
        t_pde
    )


def load_ground_truth(path="data/raw/burgers_shock.mat"):

    data = scipy.io.loadmat(path)

    x = data["x"]
    t = data["t"]
    u = data["usol"]

    return {
        "x": x,
        "t": t,
        "u": u
    }