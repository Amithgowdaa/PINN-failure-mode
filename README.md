# PINN Failure Modes

A research-focused implementation for studying **Physics-Informed Neural Network (PINN) failure modes** on partial differential equations, with an initial focus on the **1D Burgers' Equation**.

The project is designed to investigate why PINNs often struggle despite satisfying governing equations, and to analyze effects such as:

* Spectral bias
* Optimization difficulties
* Residual imbalance
* Boundary/initial condition conflicts
* Shock formation and high-frequency solution failure
* Activation function sensitivity

---

## Project Structure

```text
pinn-failure-mode/
├── conditions/
│   ├── boundary/
│   │   └── dirichlet_zero.py
│   └── initial/
│       └── burgers_ic.py
│
├── data/
│   ├── raw/
│   │   └── burgers_shock.mat
│   └── burgers_data.py
│
├── equations/
│   └── burgers_eq.py
│
├── experiments/
│   └── spectral_bias/
│       ├── history.json
│       └── model.pt
│
├── losses/
│   └── standard.py
│
├── models/
│   ├── activation.py
│   └── sequential.py
│
├── utils/
│   └── plot.py
│
├── train.py
└── README.md
```

---

## Problem Definition

We study the viscous Burgers' equation:

[
u_t + u u_x - \nu u_{xx} = 0
]

with:

### Initial Condition

[
u(x,0) = -\sin(\pi x)
]

### Boundary Conditions

[
u(-1,t)=0
]

[
u(1,t)=0
]

where:

* (x \in [-1,1])
* (t \in [0,1])
* (\nu = \frac{0.01}{\pi})

The reference solution is loaded from the standard `burgers_shock.mat` dataset.

---

## Objectives

This repository is intended to answer questions such as:

* Why do PINNs fail near shocks?
* How severe is spectral bias in PINNs?
* Which activation functions improve convergence?
* How does residual weighting affect training?
* Do lower losses correspond to physically correct solutions?
* How do collocation point distributions impact accuracy?

---

## Components

### Equation Module

`equations/burgers_eq.py`

Defines:

* Burgers equation parameters
* Automatic differentiation operators
* PDE residual computation

Residual:

```python
f = u_t + u * u_x - nu * u_xx
```

---

### Boundary Conditions

`conditions/boundary/dirichlet_zero.py`

Implements:

```python
u(-1,t) = 0
u(1,t) = 0
```

Boundary loss is computed separately and added to the total training objective.

---

### Initial Condition

`conditions/initial/burgers_ic.py`

Implements:

```python
u(x,0) = -sin(pi*x)
```

Used to constrain the solution at the initial time.

---

### Neural Network

`models/sequential.py`

Standard multilayer perceptron:

```text
(x, t)
   ↓
Fully Connected Layers
   ↓
Activation
   ↓
Output u(x,t)
```

Input:

```python
[x, t]
```

Output:

```python
u
```

---

### Supported Activations

`models/activation.py`

Available activations:

* Tanh
* ReLU
* GELU
* Swish
* Sine

This enables direct comparison of activation-induced failure modes.

---

### Loss Function

`losses/standard.py`

Total loss:

[
L =
\lambda_f L_{PDE}
+
\lambda_b L_{BC}
+
\lambda_i L_{IC}
]

where:

* (L_{PDE}): physics residual loss
* (L_{BC}): boundary condition loss
* (L_{IC}): initial condition loss

---

### Dataset

`data/burgers_data.py`

Responsibilities:

* Loading `burgers_shock.mat`
* Sampling collocation points
* Generating boundary samples
* Generating initial condition samples
* Providing evaluation grids

Place the dataset here:

```text
data/raw/burgers_shock.mat
```

---

## Training

Run:

```bash
python train.py
```

Typical workflow:

1. Load Burgers dataset
2. Generate collocation points
3. Construct PINN model
4. Compute PDE residuals
5. Optimize combined loss
6. Save training history
7. Save trained weights

Outputs:

```text
experiments/spectral_bias/
├── history.json
└── model.pt
```

---

## Visualization

`utils/plot.py`

Supported plots:

### Loss Curves

Shows convergence of:

* PDE loss
* Boundary loss
* Initial condition loss
* Total loss

### Residual Heatmap

Visualizes:

[
|f(x,t)|
]

across the domain.

### Prediction vs Ground Truth

Compares:

* Reference Burgers solution
* PINN prediction
* Absolute error

---

## Example Experiments

### Spectral Bias

Investigate whether PINNs learn:

1. Low-frequency modes first
2. High-frequency modes later
3. Shock regions last

Output:

```text
experiments/spectral_bias/
```

---

### Activation Study

Compare:

* Tanh
* ReLU
* GELU
* Swish
* Sine

Metrics:

* Final L2 error
* PDE residual
* Convergence speed
* Shock accuracy

---

### Loss Weight Sensitivity

Vary:

```python
lambda_f
lambda_b
lambda_i
```

Observe:

* Stability
* Accuracy
* Constraint satisfaction

---

## Research Questions

This repository is built to explore:

* Spectral bias in PINNs
* Gradient pathologies
* Multi-objective optimization imbalance
* Shock-learning limitations
* Activation function effects
* Sampling strategy effects
* Generalization under sparse supervision

---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E.
   *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear PDEs.*

2. Wang, S., Teng, Y., & Perdikaris, P.
   *Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks.*

3. Krishnapriyan, A. et al.
   *Characterizing Possible Failure Modes in Physics-Informed Neural Networks.*

4. Burgers' Equation Benchmark Dataset (`burgers_shock.mat`).

---

## License

MIT License

---

## Citation

```bibtex
@software{pinn_failure_modes,
  title={PINN Failure Modes},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/pinn-failure-mode}
}
```
