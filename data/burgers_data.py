import torch
import scipy.io


def load_training_data(
    N_ic:  int = 100,
    N_bc:  int = 100,
    N_pde: int = 10000,
):
    """
    Samples collocation points for Burgers' equation on:
        x ∈ [-1, 1],  t ∈ [0, 1]

    Returns a dict (not a tuple) so callers access by name, not position.
    Returning a 6-tuple is fragile — if you add a new condition later, every
    caller that unpacks positionally silently gets the wrong tensor.

    BUG FIXED: original sampled t_bc as one block of N_bc points, then
    concatenated x_bc as [left_half | right_half]. The pairing was implicit
    and ambiguous. Now t_bc is sampled in two explicit halves to match x_bc.
    Functionally equivalent for uniform sampling, but now the structure is
    explicit and safe to extend (e.g. non-uniform boundary schedules).
    """

    # --- Initial condition:  t = 0,  x ∈ [-1, 1] ---
    x_ic = torch.FloatTensor(N_ic, 1).uniform_(-1, 1)
    t_ic = torch.zeros(N_ic, 1)

    # --- Boundary condition:  x = ±1,  t ∈ [0, 1] ---
    # Left boundary  (x = -1)
    x_bc_left  = -torch.ones(N_bc // 2, 1)
    t_bc_left  =  torch.FloatTensor(N_bc // 2, 1).uniform_(0, 1)

    # Right boundary (x = +1)
    x_bc_right =  torch.ones(N_bc // 2, 1)
    t_bc_right =  torch.FloatTensor(N_bc // 2, 1).uniform_(0, 1)

    x_bc = torch.cat([x_bc_left,  x_bc_right], dim=0)
    t_bc = torch.cat([t_bc_left,  t_bc_right], dim=0)

    # --- PDE collocation:  (x, t) ∈ [-1,1] × [0,1] ---
    x_pde = torch.FloatTensor(N_pde, 1).uniform_(-1, 1)
    t_pde = torch.FloatTensor(N_pde, 1).uniform_(0,  1)

    return {
        "x_ic":  x_ic,
        "t_ic":  t_ic,
        "x_bc":  x_bc,
        "t_bc":  t_bc,
        "x_pde": x_pde,
        "t_pde": t_pde,
    }


def load_ground_truth(path: str = "data/raw/burgers_shock.mat") -> dict:
    """
    Loads the reference solution from Raissi et al. 2019.
    Download: https://github.com/maziarraissi/PINNs (data folder)

    Returns:
        x    : (256,  1)  spatial grid
        t    : (100,  1)  temporal grid
        u    : (256, 100) solution field
    """
    data = scipy.io.loadmat(path)

    return {
        "x": data["x"].flatten(),    # flatten to 1D for easier indexing
        "t": data["t"].flatten(),
        "u": data["usol"],           # shape (256, 100)
    }