import torch
import numpy as np
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict


class WarpedInputMaternKernel(gpytorch.kernels.Kernel):
    """
    Kernel wrapper that applies an input warp (atan) before computing
    a Matern kernel. Warping can help stabilize inputs with different scales.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = MaternKernel(nu=1.5)

    def forward(self, x1, x2, **params):
        # Apply a smooth monotonic warp (arctan) to inputs, then evaluate base kernel.
        x1_warped = torch.atan(x1)
        x2_warped = torch.atan(x2)
        return self.base_kernel(x1_warped, x2_warped, **params)


class SingleObjectiveGP(ExactGP):
    """
    Exact GP model for a single scalar objective using a constant mean
    and a scaled warped Matern covariance.
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(WarpedInputMaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiObjectiveGP(gpytorch.Module):
    """
    Container for independent single-objective GPs when modeling multiple objectives.
    Each objective gets its own ExactGP instance and likelihood.
    """

    def __init__(self, train_x, train_y_list, likelihoods):
        super().__init__()
        self.gps = torch.nn.ModuleList(
            [
                SingleObjectiveGP(train_x, y, lik)
                for y, lik in zip(train_y_list, likelihoods)
            ]
        )

    def forward(self, x):
        # Return a list of GP predictive distributions (one per objective).
        return [gp(x) for gp in self.gps]


class SurrogateModel:
    """
    SurrogateModel: wrapper around GPyTorch exact GPs for single- and multi-objective regression.

    This class:
      - standardizes inputs and outputs (per-objective),
      - constructs GP models (SingleObjectiveGP / MultiObjectiveGP),
      - optimizes hyperparameters via LBFGS,
      - provides predict(), save_model(), load_model() helpers.

    Notes:
      - For small datasets ExactGP is appropriate; for larger data consider sparse GPs.
      - The training loop uses a short LBFGS optimization (20 iterations).
    """

    def __init__(self, num_objectives=1):
        self.num_objectives = num_objectives
        self.models = None
        self.likelihoods = None
        self.train_x = None
        self.train_y_list = None

        # Standard scalers for inputs and outputs
        self.scaler_x = StandardScaler()
        self.scaler_y_list = [StandardScaler() for _ in range(num_objectives)]

    def fit(self, X: List[List[float]], y_list: List[List[float]]):
        """
        Fit GP(s) on provided data.

        Args:
            X: 2D array-like of input features (n_samples x n_features).
            y_list: List of target arrays, one per objective. For single objective,
                    pass a list with one array.
        """
        # Standardize and convert to torch tensors
        self.train_x = torch.tensor(self.scaler_x.fit_transform(X), dtype=torch.float32)

        # Fit per-objective scalers and convert targets
        self.train_y_list = []
        for i, y in enumerate(y_list):
            y_arr = np.array(y).reshape(-1, 1)
            y_scaled = self.scaler_y_list[i].fit_transform(y_arr).flatten()
            self.train_y_list.append(torch.tensor(y_scaled, dtype=torch.float32))

        # Create Gaussian likelihoods (one per objective)
        self.likelihoods = [GaussianLikelihood() for _ in range(self.num_objectives)]

        # Construct model(s)
        if self.num_objectives == 1:
            self.models = SingleObjectiveGP(self.train_x, self.train_y_list[0], self.likelihoods[0])
        else:
            self.models = MultiObjectiveGP(self.train_x, self.train_y_list, self.likelihoods)

        # Switch models and likelihoods to training mode
        for m in self.get_gps():
            m.train()
            # for ExactGP the likelihood is an attribute passed in init; ensure training mode
        for lik in self.likelihoods:
            lik.train()

        # Collect parameters from all GP model(s) for optimization
        params = [p for m in self.get_gps() for p in m.parameters()]

        # Use LBFGS for a small number of GP hyperparameters (good for second-order updates)
        optimizer = torch.optim.LBFGS(params, lr=0.1)

        # Prepare marginal log-likelihood(s)
        if self.num_objectives == 1:
            mll = ExactMarginalLogLikelihood(self.likelihoods[0], self.models)
        else:
            mll_list = [
                ExactMarginalLogLikelihood(lik, gp)
                for lik, gp in zip(self.likelihoods, self.models.gps)
            ]

        def closure():
            """
            Closure required by LBFGS: zero grad, compute negative log marginal likelihood, backprop, return loss.
            """
            optimizer.zero_grad()
            output = self.models(self.train_x)
            if self.num_objectives == 1:
                loss = -mll(output, self.train_y_list[0])
            else:
                # Sum negative log marginal likelihoods across objectives
                loss = -sum(mll_i(out, y) for mll_i, out, y in zip(mll_list, output, self.train_y_list))
            loss.backward()
            return loss

        # Run a small number of L-BFGS steps (adjust iterations as needed)
        for _ in range(20):
            optimizer.step(closure)

    def predict(self, X: List[List[float]]):
        """
        Predict mean and variance for given inputs.

        Args:
            X: 2D array-like of input features (n_queries x n_features).

        Returns:
            For single-objective: (mean_array, variance_array)
            For multi-objective: list of (mean_array, variance_array) tuples, one per objective.
        """
        X = np.atleast_2d(X)
        X_tensor = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32)

        # Put models and likelihoods into evaluation mode
        for m in self.get_gps():
            m.eval()
        for lik in self.likelihoods:
            lik.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.models(X_tensor)

        if self.num_objectives == 1:
            # preds is a MultivariateNormal for single objective
            mean = preds.mean.numpy()
            std = preds.variance.sqrt().numpy()
            # Inverse-transform outputs back to original scale
            mean = self.scaler_y_list[0].inverse_transform(mean.reshape(-1, 1)).flatten()
            std = std * self.scaler_y_list[0].scale_[0]
            return mean, std**2
        else:
            # preds is a list of MultivariateNormal objects
            results = []
            for i, p in enumerate(preds):
                mean = p.mean.numpy()
                std = p.variance.sqrt().numpy()
                mean = self.scaler_y_list[i].inverse_transform(mean.reshape(-1, 1)).flatten()
                std = std * self.scaler_y_list[i].scale_[0]
                results.append((mean, std**2))
            return results

    def save_model(self, filename: str):
        """
        Save model state, likelihoods, training data and scalers to disk using torch.save.
        """
        torch.save({
            'num_objectives': self.num_objectives,
            'model_state_dicts': [m.state_dict() for m in self.get_gps()],
            'likelihood_state_dicts': [l.state_dict() for l in self.likelihoods],
            'train_x': self.train_x,
            'train_y_list': self.train_y_list,
            'scaler_x': self.scaler_x,
            'scaler_y_list': self.scaler_y_list
        }, filename)

    @classmethod
    def load_model(cls, filename: str):
        """
        Load a saved surrogate model from disk and reconstruct model objects.

        Returns:
            SurrogateModel instance with loaded weights and scalers.
        """
        checkpoint = torch.load(filename, weights_only=False)
        model = cls(num_objectives=checkpoint['num_objectives'])
        model.train_x = checkpoint['train_x']
        model.train_y_list = checkpoint['train_y_list']
        model.scaler_x = checkpoint['scaler_x']
        model.scaler_y_list = checkpoint['scaler_y_list']

        # Reconstruct likelihood objects
        model.likelihoods = [GaussianLikelihood() for _ in range(model.num_objectives)]

        # Rebuild model structure and load state dicts
        if model.num_objectives == 1:
            m = SingleObjectiveGP(model.train_x, model.train_y_list[0], model.likelihoods[0])
            m.load_state_dict(checkpoint['model_state_dicts'][0])
            model.models = m
        else:
            model.models = MultiObjectiveGP(model.train_x, model.train_y_list, model.likelihoods)
            for i, m in enumerate(model.models.gps):
                m.load_state_dict(checkpoint['model_state_dicts'][i])

        # Load likelihood state dicts and set models to eval mode
        for m, lik, sd in zip(model.get_gps(), model.likelihoods, checkpoint['likelihood_state_dicts']):
            m.eval()
            lik.load_state_dict(sd)
            lik.eval()

        return model

    def get_gps(self):
        """
        Helper to return a list of GP model objects for uniform iteration.
        For single-objective returns [model], for multi-objective returns model.gps list.
        """
        return [self.models] if self.num_objectives == 1 else list(self.models.gps)
