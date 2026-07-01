
# **Conformal deformation latent space models for networks**
Carlos Nosa and Juan Sosa. Facultad de Ciencias, Universidad Nacional de Colombia.



This project proposes a new family of Bayesian latent space models in which the underlying geometry is locally deformed through node-specific multiplicative effects. Instead of assuming a homogeneous latent geometry, each node induces a local conformal scaling of the metric, producing heterogeneous effective distances while preserving the global geometry of the latent space. The proposed methodology extends classical latent distance models by allowing local metric deformations on Euclidean, spherical and hyperbolic manifolds.

## Main Contributions

The repository implements the methodology developed in the accompanying paper.

The main contributions are


1. A geometrically motivated extension of latent space models through
node-specific conformal deformations.

2. A Bayesian hierarchical formulation including uncertainty over latent
positions and deformation parameters.

3. Support for multiple latent geometries: Euclidean, Spherical and Hyperbolic.

4. Inference via Riemannian optimization.

5. Extensive posterior predictive analyses.

6. Comparative experiments with classical latent space models.


## Model

For an undirected binary network $\mathbf{Y}=(y_{i,j})$, the probability of an edge is $y_{i,j} \sim \mathrm{Bernoulli}(p_{ij})$ where $p_{ij} = \mathrm{expit}(s_{ij})$. Unlike classical latent space models, $s_{ij} = \alpha_0 - \frac{\xi_i+\xi_j}{2} \,d_\kappa(z_i,z_j)$, where:


- $z_i$ denotes the latent position,

- $\xi_i$ is the node-specific conformal deformation parameter,

- $d_\kappa$ is the geodesic distance induced by the latent geometry,

- $\kappa\in\{-C,0,C\}$ determines the sectional curvature.


The deformation parameters modify the local metric scale while preserving the global manifold structure.

## Repository Structure

```

ConformalDeformationLSMN/

│
├── Examples/
│   ├── Coauthorships/
│   ├── Conclave/
│   ├── Dolphins/
│   ├── FBPagesFood/
│   ├── FirmTech/
│   ├── Florentine/
│   ├── Karate/
│   ├── LesMiserables/
│   └── Words/
│
├── Experiments/
│   └── experiments.ipynb
│
├── Images/
│
├── Modules/
│   ├── inference.py
│   ├── latentspacemodel.py
│   └── afteranalyses.py
│
├── README.md
└── ConformalDeformationLSMN_2026.pdf

```

## Repository Components

### Examples 

Contains several benchmark network datasets used throughout the experiments,
including: Zachary's Karate Club, Florentine Families, Les Miserables, Dolphin social network, Network science coauthorships, Facebook Food Pages, Word adjacency network, Firm tech collaboration network, Conclave network.


### Modules

The implementation is organized into three main modules.

- *latentspacemodel.py*: Core implementation of the proposed conformal latent space model. Responsibilities include likelihood evaluation, posterior computation, latent geometries, conformal deformation model, simulation.

- *inference.py*: Implements estimation algorithms including optimization, Langevin proposals, Gibbs updates, Metropolis-Hastings, hybrid MCMC. 

- *afteranalyses.py*: Contains diagnostic tools including posterior  predictive checks, ROC curves, precision-recall curves, goodness-of-fit, latent space visualization, graph statistics. 


### Installation

Clone the repository:

```
git clone https://github.com/cnosa/ConformalDeformationLSMN.git
```


## Citation

If this repository contributes to your research, please cite

```
@article{Nosa2026,
title={Conformal deformation latent space models for networks},
author={Nosa, Carlos and Sosa, Juan},
year={2026},
URL={https://github.com/cnosa/ConformalDeformationLSMN}
}
```

## License

This repository is released for academic and research purposes.

