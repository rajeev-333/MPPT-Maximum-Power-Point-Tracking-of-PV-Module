# MPPT-Maximum-Power-Point-Tracking-of-PV-Module
This project implements and compares multiple algorithms to optimize the duty ratio of a DC-DC boost converter so that a PV module always operates at its Maximum Power Point (MPP).

The whole pipeline is implemented in Python and is simulation-oriented (algorithm design, analysis, and comparison).

Features:
- Physics-based PV module model (single-diode equivalent circuit)

- Boost converter model with duty–ratio to input–voltage relation

- Multiple MPPT / optimization algorithms:

- Perturb & Observe (P&O)

- Non-linear optimization (SLSQP, L-BFGS-B)

- Particle Swarm Optimization (PSO)

- Artificial Neural Network (ANN-based MPPT)

- Dynamic Programming (DP / Value Iteration)

Plots and metrics:

- PV I–V and P–V characteristics

- Algorithm convergence and tracking behavior

- Power vs duty ratio “landscape”

- ANN training curves and prediction quality


