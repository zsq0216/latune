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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = MaternKernel(nu=1.5)

    def forward(self, x1, x2, **params):
        x1_warped = torch.atan(x1)
        x2_warped = torch.atan(x2)
        return self.base_kernel(x1_warped, x2_warped, **params)


class SingleObjectiveGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(WarpedInputMaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiObjectiveGP(gpytorch.Module):
    def __init__(self, train_x, train_y_list, likelihoods):
        super().__init__()
        self.gps = torch.nn.ModuleList([
            SingleObjectiveGP(train_x, y, lik)
            for y, lik in zip(train_y_list, likelihoods)
        ])

    def forward(self, x):
        return [gp(x) for gp in self.gps]


class SurrogateModel:
    def __init__(self, num_objectives=1):
        self.num_objectives = num_objectives
        self.models = None
        self.likelihoods = None
        self.train_x = None
        self.train_y_list = None
        self.scaler_x = StandardScaler()
        self.scaler_y_list = [StandardScaler() for _ in range(num_objectives)]

    def fit(self, X, y_list):
        self.train_x = torch.tensor(self.scaler_x.fit_transform(X), dtype=torch.float32)
        self.train_y_list = []
        for i, y in enumerate(y_list):
            y_scaled = self.scaler_y_list[i].fit_transform(np.array(y).reshape(-1, 1)).flatten()
            self.train_y_list.append(torch.tensor(y_scaled, dtype=torch.float32))

        self.likelihoods = [GaussianLikelihood() for _ in range(self.num_objectives)]

        if self.num_objectives == 1:
            self.models = SingleObjectiveGP(self.train_x, self.train_y_list[0], self.likelihoods[0])
        else:
            self.models = MultiObjectiveGP(self.train_x, self.train_y_list, self.likelihoods)

        for m in self.get_gps():
            m.train()
            m.likelihood.train()

        params = [p for m in self.get_gps() for p in m.parameters()]
        optimizer = torch.optim.LBFGS(params, lr=0.1)

        if self.num_objectives == 1:
            mll = ExactMarginalLogLikelihood(self.likelihoods[0], self.models)
        else:
            mll_list = [ExactMarginalLogLikelihood(l, m) for l, m in zip(self.likelihoods, self.models.gps)]

        def closure():
            optimizer.zero_grad()
            output = self.models(self.train_x)
            if self.num_objectives == 1:
                loss = -mll(output, self.train_y_list[0])
            else:
                loss = -sum(mll_i(out, y) for mll_i, out, y in zip(mll_list, output, self.train_y_list))
            loss.backward()
            return loss

        for _ in range(20):
            # optimizer.step(closure)
            optimizer.step(closure)


    def predict(self, X):
        X = np.atleast_2d(X)
        X_tensor = torch.tensor(self.scaler_x.transform(X), dtype=torch.float32)

        for m in self.get_gps():
            m.eval()
            m.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.models(X_tensor)

        if self.num_objectives == 1:
            mean = preds.mean.numpy()
            std = preds.variance.sqrt().numpy()
            mean = self.scaler_y_list[0].inverse_transform(mean.reshape(-1, 1)).flatten()
            std = std * self.scaler_y_list[0].scale_[0]
            return mean, std**2
        else:
            results = []
            for i, p in enumerate(preds):
                mean = self.scaler_y_list[i].inverse_transform(p.mean.numpy().reshape(-1, 1)).flatten()
                std = p.variance.sqrt().numpy() * self.scaler_y_list[i].scale_[0]
                results.append((mean, std**2))
            return results



    def save_model(self, filename):
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
    def load_model(cls, filename):
        checkpoint = torch.load(filename)
        model = cls(num_objectives=checkpoint['num_objectives'])
        model.train_x = checkpoint['train_x']
        model.train_y_list = checkpoint['train_y_list']
        model.scaler_x = checkpoint['scaler_x']
        model.scaler_y_list = checkpoint['scaler_y_list']

        model.likelihoods = [GaussianLikelihood() for _ in range(model.num_objectives)]

        if model.num_objectives == 1:
            m = SingleObjectiveGP(model.train_x, model.train_y_list[0], model.likelihoods[0])
            m.load_state_dict(checkpoint['model_state_dicts'][0])
            model.models = m
        else:
            model.models = MultiObjectiveGP(model.train_x, model.train_y_list, model.likelihoods)
            for i, m in enumerate(model.models.gps):
                m.load_state_dict(checkpoint['model_state_dicts'][i])

        for m, lik, sd in zip(model.get_gps(), model.likelihoods, checkpoint['likelihood_state_dicts']):
            m.eval()
            lik.load_state_dict(sd)
            lik.eval()

        return model

    def get_gps(self):
        return [self.models] if self.num_objectives == 1 else self.models.gps

