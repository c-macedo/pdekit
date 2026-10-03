# pdekit

Finite-difference solvers for three model PDEs, written from scratch in NumPy, each
with a convergence study that **measures** the order of accuracy against a manufactured
solution.

| Equation | Type | Schemes |
|---|---|---|
| 1D heat, $u_t = \alpha u_{xx}$ | parabolic | FTCS, BTCS, Crank–Nicolson |
| 1D advection, $u_t + a u_x = 0$ | hyperbolic | upwind, Lax–Wendroff, FTCS (unstable, on purpose) |
| 2D Poisson, $-\nabla^2 u = f$ | elliptic | five-point stencil, sparse direct solve |

> Work in progress. Results and the convergence plot go here once they exist.

## Install and test

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
