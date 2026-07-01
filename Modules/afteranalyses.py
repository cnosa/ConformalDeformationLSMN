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
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score)
from sklearn.model_selection import KFold
from Modules.latentspacemodel import *
from Modules.latentspacemodel import (LatentSpaceModel, Geometry,
                              Simplex, EuclideanGeometry, HyperbolicGeometry, SphericalGeometry,
                              LatentSpaceLikelihood, LatentSpacePrior, LatentSpacePosterior)
from Modules.inference import *
from Modules.inference import BaseInference, MAPInference, MCMCInference



class ModelAnalysis:
    # Initialization
    def __init__(self, model, inference_key="MAP"):
        self.model = model
        self.inference_key = inference_key
        # inferred object
        self.results = model.inferred[inference_key]
        # observed data
        self.Y = np.array(model.Y)
        # geometry
        self.geometry = model.geometry
        # inferred quantities
        self.params = self.results["final_params"]
        self.alpha0 = self.params["alpha0"]
        self.xi = np.array(self.params["xi"])
        self.Z = np.array(self.params["Z"])
        # derived matrices
        self.Dmat = np.array(self.results["Dmat"])
        self.P = np.array(self.results["P"])
        # optimization history
        self.history = self.results["history"]
        # dimensions
        self.n = self.Y.shape[0]
        self.d = self.geometry.d


    # SECTION 1: Matrix visualizations
    def compare_with_adjacency(self, matrix=None, title=""):
        """
        Compare adjacency matrix against an inferred matrix.
        Parameters
        ----------
        matrix : ndarray or None
            Matrix to compare against Y.
            If None, uses inferred probability matrix.
        title : str
            Title for inferred matrix.
        """
        Y = np.asarray(self.Y)
        if matrix is None:
            matrix = np.asarray(self.P)
            np.fill_diagonal(matrix, 0)
            title = "Probability matrix"
        matrix = np.asarray(matrix)
        np.fill_diagonal(matrix, 0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Adjacency matrix
        im0 = axes[0].imshow(Y, cmap="Blues", vmin=0, vmax=1)
        axes[0].set_title("Adjacency matrix")
        axes[0].set_xlabel("j")
        axes[0].set_ylabel("i")
        axes[0].set_xticks(np.arange(self.n))
        axes[0].set_yticks(np.arange(self.n))
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_label("Y")
        # Inferred matrix
        im1 = axes[1].imshow(matrix, cmap="Reds", vmin=0, vmax=1)
        axes[1].set_title(title)
        axes[1].set_xlabel("j")
        axes[1].set_ylabel("i")
        axes[1].set_xticks(np.arange(self.n))
        axes[1].set_yticks(np.arange(self.n))
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label("Probability")

        plt.tight_layout()
        plt.show()

    # SECTION 2: Optimization diagnostics
    def plot_logposterior_trace(self):
        """
        Optimization convergence diagnostics.
        Shows:
        - log-posterior
        - log-likelihood
        - log-prior
        """
        logpost = np.asarray(self.history["logpost"])
        loglik = np.asarray(self.history["loglik"])
        logprior = np.asarray(self.history["logprior"])
        iters = np.arange(len(logpost))
        plt.figure(figsize=(10, 5))
        plt.plot(iters,logpost, lw=2, label="Log-posterior")
        plt.plot(iters,loglik, lw=2, label="Log-likelihood")
        plt.plot(iters, logprior, lw=2,label="Log-prior")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("Optimization diagnostics")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    # SECTION 3: Internal validation
    def _network_statistics_from_matrix(self, Y):
        """
        Compute network statistics from adjacency matrix.
        """
        Y = np.asarray(Y)
        G = nx.from_numpy_array(Y)
        stats = {}
        # density
        stats["density"] = nx.density(G)
        # transitivity
        stats["transitivity"] = nx.transitivity(G)
        # clustering
        stats["clustering"] = nx.average_clustering(G)
        # assortativity
        try:
            stats["assortativity"] = nx.degree_assortativity_coefficient(G)
        except:
            stats["assortativity"] = np.nan
        # degrees
        deg = np.array([d for _, d in G.degree()])
        stats["average_degree"] = np.mean(deg)
        stats["degree_variance"] = np.var(deg)
        # giant component
        if nx.number_connected_components(G) > 0:
            giant_nodes = max(nx.connected_components(G),key=len)
            G_gc = G.subgraph(giant_nodes).copy()
        else:
            G_gc = G
        # average shortest path
        try:
            stats["average_shortest_path"] = nx.average_shortest_path_length(G_gc)
        except:
            stats["average_shortest_path"] = np.nan
        # modularity
        try:
            communities = list(greedy_modularity_communities(G))
            stats["modularity"] = nx.algorithms.community.modularity(G,communities)
        except:
            stats["modularity"] = np.nan
        return stats
    
    def network_statistics(self):
        """
        Statistics of observed network.
        """
        return self._network_statistics_from_matrix(self.Y)
    
    def _simulate_networks(self, n_sim=100, seed=123, force=False):
        """
        Simulate replicated networks from fitted model.
        Results are cached inside model.inferred.
        """
        if (not force
            and "internal_validation" in self.model.inferred[self.inference_key]
            and "Yrep" in self.model.inferred[self.inference_key]["internal_validation"]):
            return self.model.inferred[self.inference_key]["internal_validation"]["Yrep"]
        rng = np.random.default_rng(seed)
        Yrep = []
        for _ in range(n_sim):
            Ysim = rng.binomial(1,self.P)
            np.fill_diagonal(Ysim, 0)
            Yrep.append(Ysim)
        Yrep = np.asarray(Yrep)
        return Yrep
    
    def internal_validation(self, n_sim=100, seed=123, force=False):
        """
        Complete posterior predictive validation.
        Returns
        -------
        dict
        """
        Yrep = self._simulate_networks(n_sim=n_sim, seed=seed, force=force)
        observed = self.network_statistics()
        simulated = {}
        for Ysim in Yrep:
            stats = self._network_statistics_from_matrix(Ysim)
            for key, value in stats.items():
                simulated.setdefault(key, []).append(value)
        simulated = {k: np.asarray(v) for k, v in simulated.items()}
        pvalues = {}
        for stat in observed:
            pvalues[stat] = np.mean(simulated[stat] >= observed[stat])
        results = {"observed": observed, "simulated": simulated, "pvalues": pvalues, "n_sim": n_sim }
        self.model.inferred[self.inference_key].setdefault("internal_validation",{})
        self.model.inferred[self.inference_key]["internal_validation"] = {
            "Yrep": Yrep,
            "observed": observed,
            "simulated": simulated,
            "pvalues": pvalues,
            "n_sim": n_sim}
        return results

    def posterior_predictive_check(self, n_sim=100):
        if ("internal_validation" not in self.model.inferred[self.inference_key]):
            self.internal_validation(n_sim=n_sim)
        return self.model.inferred[self.inference_key ]["internal_validation"]

    def posterior_predictive_pvalues(self, n_sim=100):
        if ("internal_validation" not in self.model.inferred[self.inference_key]):
            self.internal_validation(n_sim=n_sim)
        return self.model.inferred[self.inference_key]["internal_validation"]["pvalues"]

    def plot_ppc_distributions(self, n_sim=100):
        if ("internal_validation" not in self.model.inferred[self.inference_key] ):
            self.internal_validation(n_sim=n_sim)
        validation = self.model.inferred[self.inference_key]["internal_validation"]
        observed = validation["observed"]
        simulated = validation["simulated"]
        stats_names = list(observed.keys())
        fig, axes = plt.subplots(2,4, figsize=(16,8))
        axes = axes.ravel()
        for ax, stat in zip(axes, stats_names):
            ax.hist(simulated[stat],bins=20,alpha=0.7)
            ax.axvline(observed[stat],color="red",linewidth=2)
            ax.set_title(f"{stat}\nppp={np.mean(simulated[stat] >= observed[stat]):.3f}")
        plt.tight_layout()
        plt.show()

    def _normalized_laplacian_spectrum(self, Y):
        """
        Normalized Laplacian spectrum.
        """
        Y = np.asarray(Y)
        deg = np.sum(Y, axis=1)
        inv_sqrt_deg = np.zeros_like(deg, dtype=float)
        mask = deg > 0
        inv_sqrt_deg[mask] = (1.0 / np.sqrt(deg[mask]))
        Dinv = np.diag(inv_sqrt_deg)
        L = (np.eye(Y.shape[0]) - Dinv @ Y @ Dinv)
        eigvals = np.linalg.eigvalsh(L)
        return np.sort(eigvals)
    
    def laplacian_validation(self):
        """
        Spectral validation using normalized Laplacian.
        """
        if ("internal_validation" not in self.model.inferred[self.inference_key]):
            raise ValueError("Run internal_validation() first.")
        validation = self.model.inferred[self.inference_key ]["internal_validation"]
        Yrep = validation["Yrep"]
        observed_spec = (self._normalized_laplacian_spectrum(self.Y))
        simulated_specs = []
        distances = []
        for Ysim in Yrep:
            spec = (self._normalized_laplacian_spectrum(Ysim))
            simulated_specs.append(spec)
            distances.append(np.linalg.norm(spec - observed_spec)/(2*np.sqrt(len(spec))))
        simulated_specs = np.asarray(simulated_specs)
        distances = np.asarray(distances)
        mean_spec = np.mean(simulated_specs,axis=0)
        results = {"observed_spectrum": observed_spec,
            "simulated_spectra": simulated_specs,
            "mean_spectrum": mean_spec,
            "distance_samples": distances,
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances))}
        self.model.inferred[self.inference_key]["laplacian_validation"] = results
        return results

    # SECTION 4: Binary prediction diagnostics  
    def _binary_data(self):
        """
        Extract upper-triangular links.
        """
        idx = np.triu_indices(self.n, k=1)
        y_true = self.Y[idx]
        y_score = self.P[idx]
        return y_true, y_score
    
    def binary_prediction_metrics(self, threshold=0.5):
        """
        Binary prediction diagnostics.
        """
        y_true, y_score = self._binary_data()
        y_pred = (y_score >= threshold).astype(int)
        metrics = {"roc_auc":roc_auc_score(y_true,y_score),
            "accuracy":accuracy_score(y_true,y_pred),
            "precision":precision_score(y_true,y_pred,zero_division=0),
            "recall":recall_score(y_true,y_pred,zero_division=0),
            "f1":f1_score(y_true,y_pred,zero_division=0),
            "balanced_accuracy":balanced_accuracy_score(y_true, y_pred),
            "threshold": threshold}
        self.model.inferred[self.inference_key].setdefault("binary_prediction",{})
        self.model.inferred[self.inference_key]["binary_prediction"]["metrics"] = metrics
        return metrics
    
    def plot_roc_curve(self):
        """
        ROC curve.
        """
        y_true, y_score = self._binary_data()
        fpr, tpr, _ = roc_curve(y_true,y_score)
        auc = roc_auc_score(y_true, y_score)
        self.model.inferred[self.inference_key].setdefault("binary_prediction", {})
        self.model.inferred[self.inference_key]["binary_prediction"]["roc"] = {"fpr": fpr,"tpr": tpr,"auc": auc}
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr,lw=2,label=f"AUC = {auc:.3f}")
        plt.plot([0,1],[0,1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    def plot_precision_recall_curve(self):
        """
        Precision-Recall curve.
        """
        y_true, y_score = self._binary_data()
        precision, recall, _ = precision_recall_curve(y_true,y_score)
        ap = average_precision_score(y_true, y_score)
        self.model.inferred[self.inference_key].setdefault("binary_prediction", {})
        self.model.inferred[self.inference_key]["binary_prediction"]["pr_curve"] = {"precision": precision,"recall": recall,"average_precision": ap}
        plt.figure(figsize=(6,6))
        plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def confusion_matrix(self, threshold=0.5):
        """
        Confusion matrix.
        """
        y_true, y_score = self._binary_data()
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true,y_pred)
        self.model.inferred[self.inference_key].setdefault("binary_prediction", {})
        self.model.inferred[self.inference_key]["binary_prediction"]["confusion_matrix"] = cm
        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks([0,1], ["Pred 0","Pred 1"] )
        plt.yticks([0,1], ["True 0","True 1"])
        for i in range(2):
            for j in range(2):
                plt.text(j,i,str(cm[i,j]),ha="center",va="center")
        plt.title(f"Confusion Matrix (threshold={threshold})")
        plt.tight_layout()
        plt.show()
        return cm

    def binary_prediction_analysis(self,threshold=0.5):
        """
        Complete binary prediction analysis.
        """
        metrics = self.binary_prediction_metrics(threshold=threshold)
        y_true, y_score = self._binary_data()
        fpr, tpr, _ = roc_curve(y_true,y_score)
        auc = roc_auc_score(y_true,y_score)
        precision, recall, _ = precision_recall_curve(y_true,y_score)
        ap = average_precision_score(y_true,y_score)
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true,y_pred)
        self.model.inferred[self.inference_key]["binary_prediction"] = {
            "metrics": metrics,
            "roc": {"fpr": fpr,"tpr": tpr,"auc": auc},
            "pr_curve": {"precision": precision, "recall": recall, "average_precision": ap}, "confusion_matrix": cm}
        return self.model.inferred[self.inference_key]["binary_prediction"]

    # SECTION 5: Cross-validation
    def cross_validation(self, n_folds=5, seed=42, lr=1e-2, n_iter=5000, patience=50, n_starts=1):
        """
        Link prediction K-fold cross-validation.
        """
        Y = np.asarray(self.Y)
        idx_i, idx_j = np.triu_indices(self.n, k=1 )
        edges = np.column_stack([idx_i, idx_j])
        kf = KFold(n_splits=n_folds, shuffle=True,random_state=seed)
        auc_scores = []
        fold_results = []
        theta_full = self.model.inferred["MAP"]["final_params"]
        for fold, (train_idx, test_idx) in enumerate(kf.split(edges)):
            train_edges = edges[train_idx]
            test_edges = edges[test_idx]
            mask = np.ones_like(Y)
            for i, j in test_edges:
                mask[i, j] = 0
                mask[j, i] = 0
            model_cv = LatentSpaceModel(Y=jnp.array(Y), geometry=self.geometry, fixed_params=self.model.fixed_params, mask=jnp.array(mask))
            inference = MAPInference(model_cv, lr=lr)
            inference.fit(n_iter=n_iter, patience=patience, verbose=False, use_tqdm=True,init_params=theta_full)
            P = np.array(model_cv.inferred["MAP"]["P"])
            np.fill_diagonal(P, 0)
            y_true = np.array([Y[i, j] for i, j in test_edges])
            y_score = np.array([P[i, j] for i, j in test_edges])
            auc = roc_auc_score(y_true, y_score)
            auc_scores.append(float(auc))
            fold_results.append({"fold": fold,  "auc": float(auc), "ratio_positive": np.mean(y_true)})
        results = {"auc_scores": auc_scores, "mean_auc": float(np.mean(auc_scores)),
            "std_auc": float(np.std(auc_scores)),
            "n_folds": n_folds,
            "fold_results": fold_results}
        self.model.inferred[self.inference_key]["cross_validation"] = results
        return results
    def cross_validation_summary(self):
        """
        Cross-validation summary.
        """
        cv = self.model.inferred[self.inference_key]["cross_validation"]
        summary = {"mean_auc": cv["mean_auc"],
            "std_auc": cv["std_auc"],
            "min_auc": np.min(cv["auc_scores"]),
            "max_auc": np.max(cv["auc_scores"]),
            "n_folds": cv["n_folds"]}
        return summary
