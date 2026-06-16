import torch
import json
from pathlib import Path

from models.sequential import MLP
from equations.burgers_eq import BurgersEquation
from conditions.initial.burgers_ic import BurgersIC
from conditions.boundary.dirichlet_zero import DirichletZeroBC
from losses.standard import StandardLoss
from data.burgers_data import load_training_data


def train(config: dict) -> dict:
    """
    Baseline PINN training loop for Burgers' equation.

    Args:
        config: dict with keys —
            layers      : e.g. [2, 64, 64, 64, 1]
            activation  : "tanh" | "swish" | "sine" | ...
            nu          : viscosity (default 0.01/pi)
            N_ic, N_bc, N_pde : collocation counts
            n_epochs    : training iterations
            lr          : Adam learning rate
            lambda_ic, lambda_bc, lambda_pde : loss weights
            log_every   : print interval
            save_dir    : where to write history JSON

    Returns:
        history dict with lists: "ic", "bc", "pde", "total"
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Model ---
    model = MLP(
        layers=config["layers"],
        activation=config.get("activation", "tanh")
    ).to(device)

    # --- Equation, IC, BC, Loss ---
    equation = BurgersEquation(nu=config.get("nu", 0.01 / 3.141592653589793))
    ic       = BurgersIC()
    bc       = DirichletZeroBC()
    loss_fn  = StandardLoss(
        lambda_ic  = config.get("lambda_ic",  1.0),
        lambda_bc  = config.get("lambda_bc",  1.0),
        lambda_pde = config.get("lambda_pde", 1.0),
    )

    # --- Data (static batch — full collocation set) ---
    batch = load_training_data(
        N_ic  = config.get("N_ic",  100),
        N_bc  = config.get("N_bc",  100),
        N_pde = config.get("N_pde", 10000),
    )
    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    # --- Optimiser ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    # --- History ---
    history = {"ic": [], "bc": [], "pde": [], "total": []}

    n_epochs  = config.get("n_epochs", 10000)
    log_every = config.get("log_every", 500)

    # --- Training loop ---
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        losses = loss_fn.compute(equation, ic, bc, model, batch)
        losses["total"].backward()
        optimizer.step()

        # Log scalar values — detach before storing
        for key in history:
            history[key].append(losses[key].item())

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:>6d} | "
                f"Total {losses['total'].item():.4e} | "
                f"IC {losses['ic'].item():.4e} | "
                f"BC {losses['bc'].item():.4e} | "
                f"PDE {losses['pde'].item():.4e}"
            )

    # --- Save history and model ---
    save_dir = Path(config.get("save_dir", "experiments/spectral_bias"))
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f)

    torch.save(model.state_dict(), save_dir / "model.pt")
    print(f"\nSaved → {save_dir}/history.json  and  model.pt")

    return history, model


# ---------------------------------------------------------------------------
# Default config — edit here or override by passing a dict to train()
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "layers":      [2, 64, 64, 64, 1],
    "activation":  "tanh",
    "nu":          0.01 / 3.141592653589793,
    "N_ic":        100,
    "N_bc":        100,
    "N_pde":       10000,
    "n_epochs":    10000,
    "lr":          1e-3,
    "lambda_ic":   1.0,
    "lambda_bc":   1.0,
    "lambda_pde":  1.0,
    "log_every":   500,
    "save_dir":    "experiments/spectral_bias",
}

if __name__ == "__main__":
    train(DEFAULT_CONFIG)