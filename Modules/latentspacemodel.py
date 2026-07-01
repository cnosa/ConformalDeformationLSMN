#latentspacemodel.py

import sys
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pickle


class Geometry:
    def distance(self, zi, zj):
        raise NotImplementedError
    def distance_matrix(self, Z):
        raise NotImplementedError
    def project_to_tangent(self, z, v):
        raise NotImplementedError
    def project_to_domain(self, z):
        raise NotImplementedError
    def logaritmic_map(self, z, zp):
        raise NotImplementedError
    def exponential_map(self, z, v):
        raise NotImplementedError
    def identifiability_transform(self, Z, Z_ref):
        raise NotImplementedError
    def sample_uniform_Z(self, n:int, seed:int):
        raise NotImplementedError
    

class Simplex:
    def __init__(self, n):
        self.n = n
        self.project_to_tangent = jax.jit(self._project_to_tangent)
        self.project_to_domain = jax.jit(self._project_to_domain)
        self.exponential_map   = jax.jit(self._exponential_map)

    # Tangent projection
    def _project_to_tangent(self, xi, v):
        mean_v = jnp.mean(v)
        return v - mean_v

    # Projection to simplex
    def _project_to_domain(self, xi):
        xi = jnp.maximum(xi, 1e-12)
        return self.n * xi / jnp.sum(xi)

    # Exponential map 
    def _exponential_map(self, xi, v):
        xi_new = xi * jnp.exp(v)
        return self.n * xi_new / jnp.sum(xi_new)
    

class EuclideanGeometry(Geometry):
    def __init__(self, d, D=None):
        self.d = d
        self.D = D  
        self.distance = jax.jit(self._distance)
        self.distance_matrix = jax.jit(self._distance_matrix)
        self.project_to_tangent = jax.jit(self._project_to_tangent)
        self.project_to_domain = jax.jit(self._project_to_domain)
        self.logaritmic_map = jax.jit(self._logaritmic_map)
        self.exponential_map = jax.jit(self._exponential_map)
        self.identifiability_transform = jax.jit(self._identifiability_transform)
        self.grad_distance = jax.jit(jax.grad(self._distance, argnums=0))

    # distance 
    def _distance(self, zi, zj):
        diff = zi - zj
        return jnp.sqrt(jnp.sum(diff**2) + 1e-12)
    
    # distance matrix
    def _distance_matrix(self, Z):
        diff = Z[:, None, :] - Z[None, :, :]
        return jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)
    
    # tangent projection 
    def _project_to_tangent(self, z, v):
        return v
    
    # domain projection
    def _project_to_domain(self, z):
        if self.D is None:
            return z
        radius = self.D / 2.0
        norm = jnp.linalg.norm(z)
        factor = jnp.minimum(1.0, radius / (norm + 1e-12))
        return z * factor
    
    #logaritmic map
    def _logaritmic_map(self, z, zp):
        v = zp - z
        return v
    
    # exponential map 
    def _exponential_map(self, z, v):
        y = z + v
        return y 
    
    # identifiability
    def _identifiability_transform(self, Z, Z_ref):
        # center configurations
        Z_mean = jnp.mean(Z, axis=0, keepdims=True)
        Z_ref_mean = jnp.mean(Z_ref, axis=0, keepdims=True)
        Zc = Z - Z_mean
        Z_refc = Z_ref - Z_ref_mean
        # cross-covariance matrix
        M = Zc.T @ Z_refc
        # SVD
        U, _, Vt = jnp.linalg.svd(M, full_matrices=False)
        # optimal orthogonal transformation
        R = Vt.T @ U.T
        # aligned configuration
        Z_aligned = Zc @ R
        # recenter to reference centroid
        Z_aligned = Z_aligned + Z_ref_mean
        return Z_aligned
    
    # sampling
    def sample_uniform_Z(self, n: int, key):
        if self.D is None:
            return jax.random.normal(key, shape=(n, self.d))
        R = self.D / 2.0
        key1, key2 = jax.random.split(key)
        # random directions on sphere
        X = jax.random.normal(key1, shape=(n, self.d))
        norms = jnp.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        directions = X / norms
        # radial law for uniform volume measure in ball
        u = jax.random.uniform(key2, shape=(n, 1))
        radii = R * (u ** (1.0 / self.d))
        return directions * radii
    
class SphericalGeometry(Geometry):
    def __init__(self, d, D = jnp.pi):
        self.d = d
        self.D = D
        self.R = D / jnp.pi
        self.distance = jax.jit(self._distance)
        self.distance_matrix = jax.jit(self._distance_matrix)
        self.project_to_tangent = jax.jit(self._project_to_tangent)
        self.project_to_domain = jax.jit(self._project_to_domain)
        self.logaritmic_map = jax.jit(self._logaritmic_map)
        self.exponential_map = jax.jit(self._exponential_map)
        self.identifiability_transform = jax.jit(self._identifiability_transform)
        self.grad_distance = jax.jit(jax.grad(self._distance, argnums=0))
    
    # distance on S^d(R) 
    def _distance(self, zi, zj):
        dot = jnp.dot(zi, zj) / (self.R**2)
        dot = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        return self.R * jnp.arccos(dot)
    
    # distance matrix 
    def _distance_matrix(self, Z):
        dot = (Z @ Z.T) / (self.R**2)
        dot = jnp.clip(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        return self.R * jnp.arccos(dot)
    
    #logaritmic map
    def _logaritmic_map(self, z, zp):
        dot = jnp.clip(jnp.dot(z, zp) / (self.R**2), -1.0 + 1e-12, 1.0 - 1e-12)
        theta = jnp.arccos(dot)
        sin_theta = jnp.sin(theta)
        factor = theta / (sin_theta + 1e-12)
        v = factor * (zp - dot * z)
        return v
    
    # projection to tangent space 
    def _project_to_tangent(self, z, v):
        return v - (jnp.dot(z, v) / (self.R**2)) * z
    
    # domain projection
    def _project_to_domain(self, z):
        norm = jnp.linalg.norm(z) + 1e-12
        return self.R * z / norm
  
    
    # exponential map 
    def _exponential_map(self, z, v):
        norm_v = jnp.linalg.norm(v)
        eps = 1e-12
        coef1 = jnp.cos(norm_v / self.R)
        coef2 = jnp.sin(norm_v / self.R) / (norm_v + eps)
        y = coef1 * z + coef2 * v
        return y    
    
    # identifiability 
    def _identifiability_transform(self, Z, Z_ref):
        # cross-covariance matrix
        M = Z.T @ Z_ref
        # SVD decomposition
        U, _, Vt = jnp.linalg.svd(M, full_matrices=False)
        # optimal orthogonal matrix
        Q = Vt.T @ U.T
        # avoid reflections
        detQ = jnp.linalg.det(Q)
        correction = jnp.eye(self.d + 1)
        correction = correction.at[-1, -1].set(jnp.sign(detQ))
        Q = Vt.T @ correction @ U.T
        # rotate configuration
        Z_aligned = Z @ Q
        # numerical reprojection onto sphere
        norms = jnp.linalg.norm(Z_aligned, axis=1, keepdims=True) + 1e-12
        Z_aligned = self.R * Z_aligned / norms
        return Z_aligned
    
    #sampling
    def sample_uniform_Z(self, n: int, key):
        X = jax.random.normal(key, shape=(n, self.d + 1))
        norms = jnp.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        Z = self.R * X / norms
        return Z
    

class HyperbolicGeometry(Geometry):
    def __init__(self, d, D=None):
        self.d = d
        self.D = D
        self.R = D / jnp.pi if D is not None else 1.0
        self.distance = jax.jit(self._distance)
        self.distance_matrix = jax.jit(self._distance_matrix)
        self.project_to_tangent = jax.jit(self._project_to_tangent)
        self.project_to_domain = jax.jit(self._project_to_domain)
        self.logaritmic_map = jax.jit(self._logaritmic_map)
        self.exponential_map = jax.jit(self._exponential_map)
        self.identifiability_transform = jax.jit(self._identifiability_transform)
        self.grad_distance = jax.jit(jax.grad(self._distance, argnums=0))

    # Lorentz inner product 
    def _lorentz_inner(self, zi, zj):
        return jnp.dot(zi[:-1], zj[:-1]) - zi[-1] * zj[-1]
    
    # distance
    def _distance(self, zi, zj):
        inner = self._lorentz_inner(zi, zj)
        x = -inner / (self.R**2)
        x = jnp.clip(x, 1.0 + 1e-7, None)
        return self.R * jnp.arccosh(x)
    
    # distance matrix
    def _distance_matrix(self, Z):
        spatial = Z[:, :-1]
        time = Z[:, -1]
        inner = spatial @ spatial.T - jnp.outer(time, time)
        x = -inner / (self.R**2)
        x = jnp.maximum(x, 1.0 + 1e-7)
        return self.R * jnp.arccosh(x)
    
    # projection to tangent 
    def _project_to_tangent(self, z, v):
        # T_z H^d = {v : <z,v>_L = 0}
        inner = self._lorentz_inner(z, v)
        return v + (inner / (self.R**2)) * z
    
    # domain projection
    def _project_to_domain(self, z):
        x = z[:-1]
        spatial_norm_sq = jnp.sum(x**2)
        t = jnp.sqrt(self.R**2 + spatial_norm_sq)
        z = jnp.concatenate([x, jnp.array([t])])
        t_max = self.R * jnp.cosh(self.D / (2 * self.R))
        x = z[:-1]
        t = z[-1]
        def inside():
            return z
        def outside():
            norm_x = jnp.linalg.norm(x) + 1e-12
            new_norm = jnp.sqrt(t_max**2 - self.R**2)
            factor = new_norm / norm_x
            x_new = x * factor
            return jnp.concatenate([x_new, jnp.array([t_max])])
        return jax.lax.cond(t <= t_max, inside, outside)
    
    #logaritmic map
    def _logaritmic_map(self, z, zp):
        dot = self._lorentz_inner(z, zp)
        alpha = -dot  # debe ser ≥ 1
        omega = jnp.arccosh(alpha)
        denom = jnp.sqrt(alpha**2 - 1.0 + 1e-12)
        v = (omega / (denom + 1e-12)) * (zp + (dot / (self.R**2)) * z)
        return self._project_to_tangent(z, v)
    
    # exponential map 
    def _exponential_map(self, z, v):
        norm_v_sq = self._lorentz_inner(v, v)
        norm_v = jnp.sqrt(jnp.maximum(norm_v_sq, 1e-12))
        coef1 = jnp.cosh(norm_v / self.R)
        coef2 = jnp.sinh(norm_v / self.R) / (norm_v + 1e-12)
        z_new = coef1 * z + coef2 * v
        return z_new
    
    # identifiability 
    def _identifiability_transform(self, Z, Z_ref):
        p = self.d + 1
        # Minkowski metric
        J = jnp.eye(p)
        J = J.at[-1, -1].set(-1.0)
        # generalized covariance
        M = Z_ref.T @ (Z @ J)
        # Euclidean SVD
        U, _, Vt = jnp.linalg.svd(M, full_matrices=False)
        # candidate Lorentz transform
        Q = U @ Vt
        # enforce Lorentz condition approximately
        # Q^T J Q = J
        A = Q.T @ J @ Q
        # symmetric correction
        eigvals, eigvecs = jnp.linalg.eigh(A)
        inv_sqrt = eigvecs @ jnp.diag(1.0 / jnp.sqrt(jnp.abs(eigvals) + 1e-12)) @ eigvecs.T
        Q = Q @ inv_sqrt
        # aligned configuration
        Z_aligned = Z @ Q.T
        # numerical reprojection onto hyperboloid
        spatial = Z_aligned[:, :-1]
        spatial_norm_sq = jnp.sum(spatial**2, axis=1, keepdims=True)
        time = jnp.sqrt(self.R**2 + spatial_norm_sq)
        Z_aligned = jnp.concatenate([spatial, time], axis=1)
        return Z_aligned
    
    # sampling
    def sample_uniform_Z(self, n: int, key):
        assert self.d in [1, 2], "Exact sampler implemented for d=1 or d=2"
        assert self.D is not None, "Hyperbolic sampling requires finite D"
        R = self.R
        if self.d == 1:
            key1, key2 = jax.random.split(key)
            # random direction in {-1,+1}
            direction = jax.random.choice(key1, jnp.array([-1.0, 1.0]), shape=(n, 1))
            # uniform radius in [0, D/2]
            u = jax.random.uniform(key2, shape=(n, 1))
            r = (self.D / 2.0) * u
        elif self.d == 2:
            key1, key2 = jax.random.split(key)
            # angular direction uniformly on circle
            theta = jax.random.uniform(key1, shape=(n, 1), minval=0.0, maxval=2.0 * jnp.pi)
            direction = jnp.concatenate([jnp.cos(theta), jnp.sin(theta)],axis=1)
            # exact radial law in H^2
            alpha = 1.0
            u = jax.random.uniform(key2, shape=(n, 1))
            r = (1.0 / alpha) * jnp.arccosh(1.0 + u * (jnp.cosh(alpha * (self.D / 2.0)) - 1.0))
        norm = r / R
        cosh_term = jnp.cosh(norm)
        sinh_term = jnp.sinh(norm)
        spatial = R * sinh_term * direction
        time = R * cosh_term
        Z = jnp.concatenate([spatial, time], axis=1)
        return Z
    
class LatentSpaceLikelihood:
    def __init__(self, geometry):
        self.geometry = geometry
        self.loglikelihood = jax.jit(self._loglikelihood)
        self.grad_alpha0 = jax.jit(jax.grad(self._loglikelihood, argnums=0))
        self.grad_xi     = jax.jit(jax.grad(self._loglikelihood, argnums=1))
        self.grad_Z      = jax.jit(jax.grad(self._loglikelihood, argnums=2))
    # predictor
    def _predictor(self, alpha0, xi, Z):
        D = self.geometry.distance_matrix(Z)
        xi_i = xi[:, None]
        xi_j = xi[None, :]
        S = alpha0 - 0.5 * (xi_i + xi_j) * D
        return S
    # loglikelihood
    def _loglikelihood(self, alpha0, xi, Z, Y, mask):
        S = self._predictor(alpha0, xi, Z)
        tri_mask = jnp.triu(jnp.ones_like(Y),k=1) * mask
        loglik = Y * S - jax.nn.softplus(S)
        return jnp.sum(loglik * tri_mask)
    # gradients
    def gradients(self, alpha0, xi, Z, Y, mask):
        return {"alpha0": self.grad_alpha0(alpha0, xi, Z, Y, mask), "xi": self.grad_xi(alpha0, xi, Z, Y, mask), "Z": self.grad_Z(alpha0, xi, Z, Y, mask)}
    
class LatentSpacePrior:
    def __init__(self, geometry, sigma2_alpha0=3.0, a_xi=1.0, fixed_params:dict = {}):
        self.sigma2_alpha0 = sigma2_alpha0
        self.a_xi = a_xi
        self.geometry = geometry
        self.fixed_params = fixed_params
        self.fix_alpha0 = "alpha0" in fixed_params
        if self.fix_alpha0:
            self.alpha0_fixed = self.fixed_params.get("alpha0", None)
        self.fix_xi     = "xi" in fixed_params
        if self.fix_xi:
            self.xi_fixed = self.fixed_params.get("xi", None)
        self.fix_Z      = "Z" in fixed_params
        if self.fix_Z:
            self.Z_fixed = self.fixed_params.get("Z", None)
         
        self.log_prior_density = jax.jit(self._log_prior_density)
        self.grad_alpha0 = jax.jit(jax.grad(self._log_prior_density, argnums=0))
        self.grad_xi     = jax.jit(jax.grad(self._log_prior_density, argnums=1))
        self.grad_Z      = jax.jit(jax.grad(self._log_prior_density, argnums=2))
        

    #  log prior 
    def _log_prior_density(self, alpha0, xi, Z):
        logp = 0.0
        # alpha0 ~ N(0, sigma2)
        if not self.fix_alpha0:
            logp += -0.5 * (jnp.log(self.sigma2_alpha0) + alpha0**2 / self.sigma2_alpha0)
        # xi ~ Dirichlet 
        if not self.fix_xi:
            xi_safe = jnp.clip(xi, 1e-12, 1e6)
            logp += jnp.sum((self.a_xi - 1.0) * jnp.log(xi_safe))
        # Z ~ Uniform
        if not self.fix_Z:
            logp += 0.0
        return logp

    # gradients
    def gradients(self, alpha0, xi, Z):
        return {"alpha0": self.grad_alpha0(alpha0, xi, Z),"xi": self.grad_xi(alpha0, xi, Z),"Z": self.grad_Z(alpha0, xi, Z)}

    # sampling
    def sample(self, n, seed=42):
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, 3)
        # alpha0
        if self.fix_alpha0:
            alpha0 = self.alpha0_fixed
        else:
            alpha0 = jnp.sqrt(self.sigma2_alpha0) * jax.random.normal(keys[0])
        # xi
        if self.fix_xi:
            xi = self.xi_fixed
            assert xi.shape[0] == n, "xi hasn't correct dimension"
        else:
            a_vec = jnp.ones(n) * self.a_xi
            xi = jax.random.dirichlet(keys[1], a_vec)
            xi = n * xi
        # Z
        if self.fix_Z:
            Z = self.Z_fixed
            #assert Z.shape == (n, self.geometry.d), "Z hasn't correct dimension"
        else:
            Z = self.geometry.sample_uniform_Z(n, keys[2])
        return alpha0, xi, Z
    


class LatentSpacePosterior:
    def __init__(self, likelihood, prior):
        self.likelihood = likelihood
        self.prior = prior
        self.log_posterior = jax.jit(self._log_posterior)
        self.grad_alpha0 = jax.jit(jax.grad(self._log_posterior, argnums=0))
        self.grad_xi     = jax.jit(jax.grad(self._log_posterior, argnums=1))
        self.grad_Z      = jax.jit(jax.grad(self._log_posterior, argnums=2))
    # log posterior 
    def _log_posterior(self, alpha0, xi, Z, Y, mask):
        loglik = self.likelihood.loglikelihood(alpha0, xi, Z, Y, mask)
        logprior = self.prior.log_prior_density(alpha0, xi, Z)
        return loglik + logprior   
    # gradients
    def gradients(self, alpha0, xi, Z, Y, mask):
        return {"alpha0": self.grad_alpha0(alpha0, xi, Z, Y, mask),
            "xi":     self.grad_xi(alpha0, xi, Z, Y, mask),
            "Z":      self.grad_Z(alpha0, xi, Z, Y, mask)}
    

class LatentSpaceModel:
    def __init__(self, Y, geometry, fixed_params: dict = {}, mask=None):
        self.Y = Y
        self.geometry = geometry
        self.d = self.geometry.d
        self.fixed_params = fixed_params
        self.mask = (jnp.ones_like(Y) if mask is None else mask)
        self.prior = LatentSpacePrior(geometry = self.geometry, fixed_params = self.fixed_params)
        self.likelihood = LatentSpaceLikelihood(geometry= self.geometry)
        self.posterior = LatentSpacePosterior(likelihood=self.likelihood, prior=self.prior)
        self.weights = Simplex(n = self.Y.shape[0])
        self.latent_params = {}
        self.init_params = {}
        self.simulated = {}
        self.inferred = {}
    def get_estimation_dict(self):
        latent = ["alpha0", "xi", "Z"]
        return {k: None for k in latent if k not in self.fixed_params}
    def get_params(self):
        return {**self.latent_params, **self.fixed_params}
    def initialize_from_prior(self, n, seed=42):
        alpha0, xi, Z = self.prior.sample(n, seed)
        sampled = {"alpha0": alpha0, "xi": xi, "Z": Z}
        self.latent_params = {
            k: v for k, v in sampled.items()
            if k not in self.fixed_params
        }
        self.init_params = self.get_params().copy()
        return self.get_params()
    

    def simulate_network(self, key):
        p = self.get_params()
        alpha0 = p["alpha0"]
        xi = p["xi"]
        Z = p["Z"]
        D = self.geometry.distance_matrix(Z)
        S = alpha0 - 0.5 * (xi[:, None] + xi[None, :]) * D
        P = jax.nn.sigmoid(S)
        Y = jax.random.bernoulli(key, P)
        self.simulated = {"Z": Z, "D": D, "P": P, "Y": Y}
        return Y
    
    def log_likelihood(self):
        p = self.get_params()
        return self.likelihood.loglikelihood(
            p["alpha0"], p["xi"], p["Z"], self.Y, self.mask
        )

    def log_prior(self):
        p = self.get_params()
        return self.prior.log_prior_density(
            p["alpha0"], p["xi"], p["Z"]
        )

    def log_posterior(self):
        p = self.get_params()
        return self.posterior.log_posterior(
            p["alpha0"], p["xi"], p["Z"], self.Y, self.mask
        )

    def grad_log_likelihood(self):
        p = self.get_params()
        grads = self.likelihood.gradients(p["alpha0"], p["xi"], p["Z"], self.Y, self.mask)
        if "Z" in grads:
            grads["Z"] = jax.vmap(self.geometry.project_to_tangent)(p["Z"], grads["Z"])
        return {k: grads[k] for k in self.latent_params}
    
    def grad_log_prior(self):
        p = self.get_params()
        grads = self.prior.gradients(p["alpha0"], p["xi"], p["Z"])
        if "Z" in grads:
            grads["Z"] = jax.vmap(self.geometry.project_to_tangent)(p["Z"], grads["Z"])
        return {k: grads[k] for k in self.latent_params}
    
    def grad_log_posterior(self):
        p = self.get_params()
        grads = self.posterior.gradients(p["alpha0"], p["xi"], p["Z"], self.Y, self.mask)
        if "Z" in grads:
            grads["Z"] = jax.vmap(self.geometry.project_to_tangent)(p["Z"], grads["Z"])
        return {k: grads[k] for k in self.latent_params}
    

    def to_dict(self):
        return {"Y": np.array(self.Y),
            "geometry_class": type(self.geometry).__name__,
            "geometry_params": {"d": self.geometry.d, "D": getattr(self.geometry, "D", None)},
            "fixed_params": self.fixed_params,
            "latent_params": self.latent_params,
            "init_params": self.init_params,
            "simulated": self.simulated,
            "inferred": self.inferred}
    
    def save(self, filename):
        state = self.to_dict()
        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            state = pickle.load(f)
        geometry_name = state["geometry_class"]
        if geometry_name == "EuclideanGeometry":
            geometry = EuclideanGeometry(**state["geometry_params"])
        elif geometry_name == "SphericalGeometry":
            geometry = SphericalGeometry(**state["geometry_params"])
        elif geometry_name == "HyperbolicGeometry":
            geometry = HyperbolicGeometry(**state["geometry_params"])
        else:
            raise ValueError(f"Unknown geometry: {geometry_name}")
        model = cls(Y=jnp.array(state["Y"]), geometry=geometry, fixed_params=state["fixed_params"])
        model.latent_params = state["latent_params"]
        model.init_params = state["init_params"]
        model.simulated = state["simulated"]
        model.inferred = state["inferred"]
        return model


