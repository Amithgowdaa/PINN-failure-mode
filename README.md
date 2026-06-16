# PINN-failure-mode
This notebook and test file defines how PINN fails over equations. 
pinn-failure-mode/
├── conditions/
│   ├── boundary/
│   │   └── dirichlet_zero.py      # u(±1, t) = 0
│   └── initial/
│       └── burgers_ic.py          # u(x, 0) = -sin(πx)
├── data/
│   ├── raw/                       # place burgers_shock.mat here
│   └── burgers_data.py            # collocation sampler + .mat loader
├── equations/
│   └── burgers_eq.py              # BurgersEquation class + residual fn
├── experiments/
│   └── spectral_bias/             # history.json + model.pt saved here
├── losses/
│   └── standard.py                # StandardLoss with lambda weights
├── models/
│   ├── activation.py              # Tanh, ReLU, GELU, Swish, Sine
│   └── sequential.py              # MLP (x,t) → u
├── utils/
│   └── plot.py                    # loss curves, residual heatmap, pred vs truth
├── train.py                       # main entry point
└── README.md