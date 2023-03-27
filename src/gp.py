import matplotlib.pyplot as plt
import os
from os.path import join

import hydra
from hydra.utils import log

import torch
import gpytorch

from data import load_data

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

@hydra.main(config_path="config", config_name="common")
def train_model(cfg):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(cfg)
    
    if not cfg.experiment.use_validation:
        X_train = torch.cat((X_train, X_val))
        y_train = torch.cat((y_train, y_val))
    
    X_train = X_train.to(cfg.experiment.device)
    y_train = y_train.to(cfg.experiment.device)
    X_test = X_test.to(cfg.experiment.device)
    y_test = y_test.to(cfg.experiment.device)
    
    if cfg.experiment.kernel_type == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=X_train.shape[1])
    elif cfg.experiment.kernel_type == "matern32":
        base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=X_train.shape[1])
    else:
        log.error(f"Kernel {cfg.experiment.kernel_type} not implemented")
        raise NotImplementedError(f"Kernel {cfg.experiment.kernel_type} not implemented")
        
    kernel = gpytorch.kernels.ScaleKernel(base_kernel)
    model = ExactGP(X_train, y_train, gpytorch.likelihoods.GaussianLikelihood(), kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(cfg.experiment.device)
    
    mll.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)
    
    losses = []
    for i in range(cfg.experiment.n_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        losses.append(loss.item())
        log.info(f"Epoch {i+1}/{cfg.experiment.n_epochs} - Loss: {losses[-1]}")
        optimizer.step()
        scheduler.step(losses[-1])
        
    mll.eval()
    pred_dist = model(X_test)
    rmse = gpytorch.metrics.mean_squared_error(pred_dist, y_test).sqrt()
    msll = gpytorch.metrics.mean_standardized_log_loss(pred_dist, y_test)
    
    # log results to hydra
    log.info(f"RMSE: {rmse.item()}")
    log.info(f"MSLL: {msll.item()}")
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(join(os.getcwd(), "loss.png"))
    
if __name__ == "__main__":
    train_model()