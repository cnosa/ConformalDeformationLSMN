
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

from Modules.latentspacemodel import *
from Modules.latentspacemodel import (LatentSpaceModel, Geometry,
                              Simplex, EuclideanGeometry, HyperbolicGeometry, SphericalGeometry,
                              LatentSpaceLikelihood, LatentSpacePrior, LatentSpacePosterior)

class BaseInference:
    def __init__(self, model: LatentSpaceModel):
        self.model = model
        self.history = {
            "params": [],
            "logpost": [],
            "loglik": [],
            "logprior": [],
            "grads": [],
            "step_norm": [],
        }
        self.result = None

    def fit(self, **kwargs):
        raise NotImplementedError

    def step(self, **kwargs):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError
    
class MAPInference(BaseInference): 
    def __init__(self, model, lr=1e-2):
        super().__init__(model)
        self.lr = lr
        self.model.inference_method = "MAP"
        self.model.inferred = {"MAP": {}}

    def _save_state(self, grads=None):
        params = self.model.get_params()
        logpost = self.model.log_posterior()
        loglik = self.model.log_likelihood()
        logprior = self.model.log_prior()
        self.history["params"].append({
            k: v.copy() if hasattr(v, "copy") else v
            for k, v in params.items()
        })

        self.history["logpost"].append(float(logpost))
        self.history["loglik"].append(float(loglik))
        self.history["logprior"].append(float(logprior))

        if grads is not None:
            self.history["grads"].append({
                k: v.copy() if hasattr(v, "copy") else v
                for k, v in grads.items()
            })

    def _store_in_model(self):
        final_params = {k: v.copy() if hasattr(v, "copy") else v
            for k, v in self.result.items()}
        alpha0 = final_params["alpha0"]
        xi = final_params["xi"]
        Z = final_params["Z"]
        # distance matrix
        Dmat = self.model.geometry.distance_matrix(Z)
        # score matrix
        S = alpha0 - 0.5 * (xi[:, None] + xi[None, :]) * Dmat
        # probability matrix
        P = jax.nn.sigmoid(S)
        # information criteria
        loglik = float(self.history["loglik"][-1])
        n = self.model.Y.shape[0]
        N = n * (n - 1) / 2
        k = 0
        if "alpha0" not in self.model.fixed_params:
            k += 1
        if "xi" not in self.model.fixed_params:
            k += n
        if "Z" not in self.model.fixed_params:
            k += n * self.model.geometry.d 
        AIC = -2.0 * loglik + 2.0 * k
        BIC = -2.0 * loglik + k * np.log(N)

        self.model.inferred["MAP"] = {
            # final inferred parameters
            "final_params": final_params,
            # optimization history
            "history": {
                k: v.copy() if hasattr(v, "copy") else v
                for k, v in self.history.items()
            },
            # objective values
            "final_logpost": (
                self.history["logpost"][-1]
                if self.history["logpost"]
                else None
            ),
            "final_loglik": (
                self.history["loglik"][-1]
                if self.history["loglik"]
                else None
            ),
            "final_logprior": (
                self.history["logprior"][-1]
                if self.history["logprior"]
                else None
            ),
            # geometry information
            "geometry": type(self.model.geometry).__name__,
            "dimension": self.model.geometry.d,
            "diameter": getattr(self.model.geometry, "D", None),
            # optimization settings
            "lr": self.lr,
            # derived quantities
            "Dmat": Dmat,
            "S": S,
            "P": P,
            # information criteria
            "AIC": float(AIC),
            "BIC": float(BIC),
            "n_parameters": int(k),
            "n_observations": int(N),
        }

    def step(self):
        eps = 0.01
        grads = self.model.grad_log_posterior() 
        params = self.model.latent_params
        geom = self.model.geometry
        step_norm_sq = 0.0

        # alpha0
        if "alpha0" in params:
            g_alpha0 = grads["alpha0"] / (jnp.abs(grads["alpha0"]) + eps)
            params["alpha0"] = params["alpha0"] + self.lr * g_alpha0
            step_norm_sq += float((self.lr * g_alpha0)**2)

        # xi
        if "xi" in params:
            xi = params["xi"]
            gxi = grads["xi"]
            simplex = self.model.weights
            gxi = simplex.project_to_tangent(xi, gxi)
            gxi = gxi / (jnp.linalg.norm(gxi) + eps)
            xi_new = simplex.exponential_map(xi, self.lr * gxi)
            xi_new = simplex.project_to_domain(xi_new)
            params["xi"] = xi_new
            step_norm_sq += float(jnp.sum((self.lr * gxi)**2))
            
        # Z
        if "Z" in params:
            Z = params["Z"]
            gZ = grads["Z"]
            gZ_tangent = jax.vmap(geom.project_to_tangent)(Z, gZ)
            gZ_tangent = gZ_tangent / (jnp.linalg.norm(gZ_tangent) + eps)
            Z_new = jax.vmap(geom.exponential_map)(Z, self.lr * gZ_tangent)
            Z_new = jax.vmap(geom.project_to_domain)(Z_new)
            params["Z"] = Z_new
            step_norm_sq += float(jnp.sum((self.lr * gZ_tangent)**2))
        
        self.history["step_norm"].append(step_norm_sq**0.5)
        self._save_state(grads)

    def fit(self, n_iter=5000, tol=1e-5, patience=201, verbose=True, use_tqdm=True, init_params=None):
        if init_params is not None:
            self.model.latent_params = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in init_params.items()
                if k not in self.model.fixed_params}
        else:
            self.model.initialize_from_prior(n=self.model.Y.shape[0])
        best = -jnp.inf
        best_params = None
        counter = 0
        iterator = tqdm(range(n_iter), desc="MAP") if use_tqdm else range(n_iter)
        for i in iterator:
            self.step()
            current = self.history["logpost"][-1]
            if use_tqdm:
                iterator.set_postfix({
                    "logpost": f"{current:.4f}",
                    "loglik": f"{self.history['loglik'][-1]:.4f}",
                    "logprior": f"{self.history['logprior'][-1]:.4f}",
                    "||grad||": f"{self.history["step_norm"][-1]:.4f}",
                    "patience": counter
                })
            if (current > best + tol):
                best = current
                best_params = {k: (v.copy() if hasattr(v, "copy") else v) for k,v in self.model.get_params().items()}
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                if verbose:
                    msg = f"[STOP] Early stopping at iter {i-1} (no improvement in {patience} steps)"
                    if use_tqdm:
                        iterator.write(msg)
                    else:
                        print(msg)
                break
        self.model.latent_params = best_params
        self.result = best_params
        self._store_in_model()
        return self.result
    
    def fit_multi_start(self, n_starts=25, n_iter=5000, tol=1e-5, patience=101, seeds=None, verbose=True, use_tqdm=True,init_params=None):
        best_logpost = -jnp.inf
        best_result = None
        all_runs = []
        if seeds is None:
            seeds = list(range(n_starts)) 

        print(f"Multi_start ({n_starts})")
        outer_iter = range(n_starts)
        for k in outer_iter:
            seed = seeds[k]
            if init_params is not None:
                self.model.latent_params = {kk: (vv.copy() if hasattr(vv, "copy") else vv) for kk, vv in init_params.items()
                    if kk not in self.model.fixed_params}
            else:
                self.model.initialize_from_prior(n=self.model.Y.shape[0], seed=seed)
            self.history = {"params": [], "logpost": [], "loglik": [], "logprior": [], "grads": [], "step_norm": []}
            self.fit(n_iter=n_iter, tol=tol, patience=patience, verbose=verbose, use_tqdm=use_tqdm, init_params=init_params)
            final_logpost = self.history["logpost"][-1]
            result = self.model.get_params()
            all_runs.append({"seed": seed, "logpost": final_logpost, "params": result})
            if final_logpost > best_logpost:
                best_logpost = final_logpost
                best_result = result
        self.result = best_result
        self.all_runs = all_runs
        if verbose:
            print(f"\nBest log-posterior: {best_logpost:.4f}")
        self._store_in_model()
        return best_result

    def summary(self):
        return {
            "final_params": self.result,
            "final_logpost": self.history["logpost"][-1] if self.history["logpost"] else None,
            "final_loglik": self.history["loglik"][-1] if self.history["loglik"] else None,
            "final_logprior": self.history["logprior"][-1] if self.history["logprior"] else None
        }
    

class MCMCInference(BaseInference):
    def __init__(self, model, eps_alpha0=1e-2, eps_xi=1e-2, eps_Z=1e-2, target_accept_alpha0=0.57, target_accept_xi=0.57, target_accept_Z=0.57):
        super().__init__(model)
        self.eps_alpha0 = eps_alpha0
        self.eps_xi = eps_xi
        self.eps_Z = eps_Z
        self.target_alpha0 = target_accept_alpha0
        self.target_xi = target_accept_xi
        self.target_Z = target_accept_Z
        self.geometry = self.model.geometry
        self.model.inference_method = "MCMC"
    # Log q
    def _log_q(self, x_new, x, grad_x, eps):
        mean = x + 0.5 * eps**2 * grad_x
        diff = x_new - mean
        return -0.5 * jnp.sum(diff**2) / (eps**2)

    def _log_q_manifold(self, z_new, z, grad_z, eps):
        v = self.geometry.logaritmic_map(z, z_new)
        mean = 0.5 * eps**2 * grad_z
        diff = v - mean
        return -0.5 * jnp.sum(diff**2) / (eps**2)

    # Adapt step size (Robbins–Monro)
    def _adapt(self, eps, accept, target, t):
        gamma = (t + 1) ** (-0.6)
        log_eps = jnp.log(eps) + gamma * (accept - target)
        return jnp.exp(log_eps)

    # Steps
    def step_alpha0(self, key):
        eps = self.eps_alpha0 #tau
        p = self.model.get_params()
        alpha0 = p["alpha0"]
        grad = self.model.grad_log_posterior()["alpha0"]
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1)
        prop = alpha0 + 0.5 * eps**2 * grad + eps * noise
        logp_current = self.model.log_posterior()
        self.model.latent_params["alpha0"] = prop
        logp_prop = self.model.log_posterior()
        grad_prop = self.model.grad_log_posterior()["alpha0"]
        log_q_f = self._log_q(prop, alpha0, grad, eps)
        log_q_b = self._log_q(alpha0, prop, grad_prop, eps)
        log_accept = logp_prop + log_q_b - logp_current - log_q_f
        if jnp.log(jax.random.uniform(key2)) < log_accept:
            return 1.0
        else:
            self.model.latent_params["alpha0"] = alpha0
            return 0.0

    def step_xi(self, key):
        eps = self.eps_xi
        p = self.model.get_params()
        xi = p["xi"]
        grad = self.model.grad_log_posterior()["xi"]
        simplex = self.model.weights
        grad = simplex.project_to_tangent(xi, grad)
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, shape=xi.shape)
        noise = simplex.project_to_tangent(xi, noise)
        v = 0.5 * eps**2 * grad + eps * noise
        xi_prop = simplex.exponential_map(xi, v)
        xi_prop = simplex.project_to_domain(xi_prop)
        logp_current = self.model.log_posterior()
        self.model.latent_params["xi"] = xi_prop
        logp_prop = self.model.log_posterior()
        grad_prop = self.model.grad_log_posterior()["xi"]
        grad_prop = simplex.project_to_tangent(xi_prop, grad_prop)
        log_q_f = self._log_q(xi_prop, xi, grad, eps)
        log_q_b = self._log_q(xi, xi_prop, grad_prop, eps)
        log_accept = logp_prop + log_q_b - logp_current - log_q_f
        if jnp.log(jax.random.uniform(key2)) < log_accept:
            return 1.0
        else:
            self.model.latent_params["xi"] = xi
            return 0.0
        
    def step_Z(self, key):
        eps = self.eps_Z
        geom = self.model.geometry
        Z = self.model.get_params()["Z"]
        n = Z.shape[0]
        keys = jax.random.split(key, n)
        accepted = 0.0
        for i in range(n):
            zi = Z[i]
            grad = self.model.grad_log_posterior()["Z"][i]
            grad = geom.project_to_tangent(zi, grad)
            logp_current = self.model.log_posterior()
            key1, key2 = jax.random.split(keys[i])
            noise = jax.random.normal(key1, shape=zi.shape)
            noise = geom.project_to_tangent(zi, noise)
            v = 0.5 * eps**2 * grad + eps * noise
            zi_prop = geom.exponential_map(zi, v)
            zi_prop = geom.project_to_domain(zi_prop)
            Z_prop = Z.at[i].set(zi_prop)
            self.model.latent_params["Z"] = Z_prop
            logp_prop = self.model.log_posterior()
            grad_prop = self.model.grad_log_posterior()["Z"][i]
            grad_prop = geom.project_to_tangent(zi_prop, grad_prop)
            log_q_f = self._log_q_manifold(zi_prop, zi, grad, eps)
            log_q_b = self._log_q_manifold(zi, zi_prop, grad_prop, eps)
            log_accept = logp_prop + log_q_b - logp_current - log_q_f
            if jnp.log(jax.random.uniform(key2)) < log_accept:
                Z = Z_prop
                accepted += 1.0
            else:
                self.model.latent_params["Z"] = Z
        self.model.latent_params["Z"] = Z
        return accepted / n

    def fit(self, n_samples=1000, burnin=1000, thin=1, seed=0, verbose=True, Z_ref=None, init_params={}):
        if init_params:
            self.model.latent_params = {k: v for k, v in init_params.items() if k not in self.model.fixed_params}
        else:
            self.model.initialize_from_prior(n=self.model.Y.shape[0], seed=seed)
        key = jax.random.PRNGKey(seed)
        total_iters = burnin + n_samples * thin
        samples = []
        self.history = {"logpost": [], "loglik": [], "logprior": [],
                        "eps": {"alpha0": [], "xi": [], "Z": []}, 
                        "acceptance": {"alpha0": [], "xi": [], "Z": []}}
        self.Z_ref = Z_ref
        self._Z_ref_initialized = Z_ref is not None
        iterator = tqdm(range(total_iters), desc="MCMC") if verbose else range(total_iters)
        self.sample_logpost = []
        for t in iterator:
            key, subkey = jax.random.split(key)
            k1, k2, k3 = jax.random.split(subkey, 3)
            acc_a = self.step_alpha0(k1) if "alpha0" in self.model.latent_params else 1.0
            acc_x = self.step_xi(k2) if "xi" in self.model.latent_params else 1.0
            acc_z = self.step_Z(k3) if "Z" in self.model.latent_params else 1.0

            if (not self._Z_ref_initialized) and (t == burnin):
                self.Z_ref = self.model.get_params()["Z"]
                self._Z_ref_initialized = True

            self.history["acceptance"]["alpha0"].append(float(acc_a))
            self.history["acceptance"]["xi"].append(float(acc_x))
            self.history["acceptance"]["Z"].append(float(acc_z))

            if t < burnin:
                self.eps_alpha0 = self._adapt(self.eps_alpha0, acc_a, self.target_alpha0, t)
                self.eps_xi     = self._adapt(self.eps_xi, acc_x, self.target_xi, t)
                self.eps_Z      = self._adapt(self.eps_Z, acc_z, self.target_Z, t)
            loglik = self.model.log_likelihood()
            logprior = self.model.log_prior()
            logpost = loglik + logprior  #self.model.log_posterior()
            self.history["loglik"].append(float(loglik))
            self.history["logprior"].append(float(logprior))
            self.history["logpost"].append(float(logpost))
            self.history["eps"]["alpha0"].append(float(self.eps_alpha0))
            self.history["eps"]["xi"].append(float(self.eps_xi))
            self.history["eps"]["Z"].append(float(self.eps_Z))


            if t >= burnin and ((t - burnin) % thin == 0):
                sample = {k: v.copy() if hasattr(v, "copy") else v for k, v in self.model.get_params().items()}
                if self._Z_ref_initialized:
                    sample["Z"] = self.geometry.identifiability_transform(sample["Z"], self.Z_ref)
                samples.append(sample)
                self.sample_logpost.append(float(logpost))
            if verbose:
                acc_total = 0.0
                acc_total += np.mean(np.array(self.history["acceptance"]["alpha0"]))
                acc_total += np.mean(np.array(self.history["acceptance"]["xi"]))
                acc_total += np.mean(np.array(self.history["acceptance"]["Z"]))
                acc_total /= 3
                iterator.set_postfix({"logpost": f"{logpost:.3f}", "loglik": f"{loglik:.3f}", "logprior": f"{logprior:.3f}",
                                       "acceptance": f"{acc_total:.3f}"})

        self.samples = samples
        self.compute_mcmc_diagnostics()
        self._store_in_model()
        return samples
    
    def compute_waic(self):
        Y = self.model.Y
        loglik_samples = []
        for sample in self.samples:
            alpha0 = sample["alpha0"]
            xi = sample["xi"]
            Z = sample["Z"]
            Dmat = self.model.geometry.distance_matrix(Z)
            S = alpha0 - 0.5 * (xi[:, None] + xi[None, :]) * Dmat
            P = jax.nn.sigmoid(S)
            P = jnp.clip(P, 1e-12,1.0 - 1e-12)
            loglik_ij = Y * jnp.log(P) + (1.0 - Y) * jnp.log(1.0 - P)
            loglik_samples.append(loglik_ij)
        L = jnp.stack(loglik_samples, axis=0)
        lppd = jnp.sum(jnp.log(jnp.mean(jnp.exp(L),axis=0)))
        p_waic = jnp.sum(jnp.var(L, axis=0,ddof=1))
        waic = -2.0 * (lppd - p_waic)
        return float(waic)

    def _acf(self, x, max_lag=None):
        x = np.asarray(x, dtype=float)
        n = len(x)
        if max_lag is None:
            max_lag = min(n // 2, 100)
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-12:
            return np.ones(max_lag + 1)
        acf = np.empty(max_lag + 1)
        for k in range(max_lag + 1):
            acf[k] = np.sum(x[:n-k] * x[k:]) / ((n-k) * var)
        return acf
    
    def _ess(self, x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        acf = self._acf(x)
        s = 0.0
        for rho in acf[1:]:
            if rho <= 0:
                break
            s += rho
        return n / (1.0 + 2.0 * s)

    def _mcse_mean(self, x):
        x = np.asarray(x, dtype=float)
        ess = self._ess(x)
        return np.sqrt(np.var(x, ddof=1) / ess)
    
    def compute_mcmc_diagnostics(self):
        diagnostics = {"acf": {}, "ess": {}, "mcse": {}, "summary": {}}
        # alpha0
        if "alpha0" in self.samples[0]:
            chain = np.array([float(s["alpha0"]) for s in self.samples])
            diagnostics["acf"]["alpha0"] = self._acf(chain)
            diagnostics["ess"]["alpha0"] = self._ess(chain)
            diagnostics["mcse"]["alpha0"] = self._mcse_mean(chain)
            diagnostics["summary"]["alpha0"] = diagnostics["ess"]["alpha0"]
            diagnostics["summary"]["mcse"] = diagnostics["mcse"]["alpha0"]
        # xi
        if "xi" in self.samples[0]:
            xi_chain = np.array([np.asarray(s["xi"]) for s in self.samples])
            p = xi_chain.shape[1]
            diagnostics["acf"]["xi"] = []
            diagnostics["ess"]["xi"] = np.zeros(p)
            diagnostics["mcse"]["xi"] = np.zeros(p)
            for j in range(p):
                chain = xi_chain[:, j]
                diagnostics["acf"]["xi"].append(self._acf(chain))
                diagnostics["ess"]["xi"][j] = self._ess(chain)
                diagnostics["mcse"]["xi"][j] = self._mcse_mean(chain)
            diagnostics["acf"]["xi"] = np.array(diagnostics["acf"]["xi"])
            diagnostics["summary"]["xi"] = np.min(diagnostics["ess"]["xi"])
            diagnostics["summary"]["mcse"] = np.max(diagnostics["mcse"]["xi"])
        # Z
        if "Z" in self.samples[0]:
            Z_chain = np.array([np.asarray(s["Z"]) for s in self.samples])
            n_nodes = Z_chain.shape[1]
            dplus1 = Z_chain.shape[2]
            diagnostics["acf"]["Z"] = np.empty((n_nodes, dplus1), dtype=object)
            diagnostics["ess"]["Z"] = np.zeros((n_nodes, dplus1))
            diagnostics["mcse"]["Z"] = np.zeros((n_nodes, dplus1))
            for i in range(n_nodes):
                for j in range(dplus1):
                    chain = Z_chain[:, i, j]
                    diagnostics["acf"]["Z"][i, j] = self._acf(chain)
                    diagnostics["ess"]["Z"][i, j] = self._ess(chain)
                    diagnostics["mcse"]["Z"][i,j] = (self._mcse_mean(chain))
            diagnostics["summary"]["Z"] = np.min(diagnostics["ess"]["Z"])
            diagnostics["summary"]["mcse"] = np.max(diagnostics["mcse"]["Z"])
        
        self.diagnostics = diagnostics
        return diagnostics

    def _store_in_model(self):
        diagnostics = self.diagnostics
        # Posterior mean estimate
        posterior_mean = {}
        # alpha0
        posterior_mean["alpha0"] = jnp.mean(jnp.array([s["alpha0"] for s in self.samples]))
        # xi
        xi_mean = jnp.mean(jnp.stack([s["xi"] for s in self.samples]), axis=0)
        xi_mean = self.model.weights.project_to_domain(xi_mean)
        posterior_mean["xi"] = xi_mean
        # Z
        Z_mean = jnp.mean(jnp.stack([s["Z"] for s in self.samples]), axis=0)
        Z_mean = jax.vmap(self.model.geometry.project_to_domain)(Z_mean)
        posterior_mean["Z"] = Z_mean
        # MAP estimate
        map_idx = np.argmax(self.sample_logpost)
        posterior_map = {k: (self.samples[map_idx][k].copy() if hasattr(self.samples[map_idx][k], "copy") else self.samples[map_idx][k])
            for k in self.samples[map_idx]}
        # Matrices from posterior mean
        alpha0_cm = posterior_mean["alpha0"]
        xi_cm = posterior_mean["xi"]
        Z_cm = posterior_mean["Z"]
        Dmat_cm = self.model.geometry.distance_matrix(Z_cm)
        S_cm = alpha0_cm - 0.5 * (xi_cm[:, None] + xi_cm[None, :]) * Dmat_cm
        P_cm = jax.nn.sigmoid(S_cm)
        # Matrices from MAP
        alpha0_map = posterior_map["alpha0"]
        xi_map = posterior_map["xi"]
        Z_map = posterior_map["Z"]
        Dmat_map = self.model.geometry.distance_matrix(Z_map)
        S_map = alpha0_map - 0.5 * (xi_map[:, None] + xi_map[None, :]) * Dmat_map
        P_map = jax.nn.sigmoid(S_map)
        # Information criterion
        waic = self.compute_waic()
        # Store
        self.model.inferred["MCMC"] = {
            # posterior samples
            "samples": self.samples,
            "sample_logpost": self.sample_logpost,
            # posterior mean
            "posterior_mean": posterior_mean,
            # MAP
            "posterior_map": posterior_map,
            # diagnostics
            "diagnostics": diagnostics,
            # information criterion
            "waic": waic,
            # history
            "history": self.history,
            # tuning parameters
            "eps": {"alpha0": self.eps_alpha0, "xi": self.eps_xi, "Z": self.eps_Z },
            # acceptance rates
            "acceptance": {"alpha0": self.history["acceptance"]["alpha0"],
                "xi": self.history["acceptance"]["xi"],
                "Z": self.history["acceptance"]["Z"],
                "alpha0_mean": np.mean(self.history["acceptance"]["alpha0"]),
                "xi_mean": np.mean(self.history["acceptance"]["xi"]),
                "Z_mean": np.mean(self.history["acceptance"]["Z"])},
            # geometry information
            "geometry": type(self.model.geometry).__name__,
            "dimension": self.model.geometry.d,
            "diameter": getattr(self.model.geometry,"D",None),
            # CM matrices
            "Dmat_cm": Dmat_cm,
            "S_cm": S_cm,
            "P_cm": P_cm,
            # MAP matrices
            "Dmat_map": Dmat_map,
            "S_map": S_map,
            "P_map": P_map}