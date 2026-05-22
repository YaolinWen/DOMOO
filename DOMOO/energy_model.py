from collections import defaultdict
import torch
import torch.nn as nn
import torch.autograd as autograd
import random
from tqdm import tqdm
import os
import torch.nn.functional as F
import numpy as np

def sample_langevin(x_neg, dhs_model, stepsize=0.01, n_steps=50, noise=True):
    for i in range(n_steps):
        if noise:
            x_noise = torch.randn_like(x_neg).detach()
            x_noise.normal_()
            x_neg = x_neg + 0.001 * (n_steps - i - 1) / n_steps * x_noise
        else:
            x_neg = x_neg

        x_neg.requires_grad = True
        out = dhs_model.forward(x_neg)

        x_neg_grad = torch.autograd.grad([out.sum()], [x_neg])[0] 
        x_neg = x_neg - stepsize * x_neg_grad 
        x_neg = x_neg.detach()

    return x_neg

class RiskSuppressionFactor(object):
    def __init__(self, energy_min, energy_max, init_m):
        self.min = energy_min
        self.max = energy_max
        self.init_m = init_m

    def normalize(self, energy):
        if isinstance(energy, torch.Tensor):
            max_val = torch.tensor(self.max, dtype=torch.dtype(energy)).to(energy.device)
            min_val = torch.tensor(self.min, dtype=torch.dtype(energy)).to(energy.device)
        else:
            max_val = self.max
            min_val = self.min
        
        normalized_energy = (max_val - energy) / (max_val - min_val + 1e-8) 
        return normalized_energy

class DualHeadSurogateTrainer(object):
    def __init__(self, dhs_model,
                 surrogate_model,
                 dhs_model_energy_opt=torch.optim.Adam,
                 surrogate_lr=0.001,
                 init_m=0.05, ldk=50):

        self.dhs_model = dhs_model
        self.surrogate_model = surrogate_model
        self.dhs_model_energy_opt = dhs_model_energy_opt(self.dhs_model.parameters(), lr=surrogate_lr)

        # algorithm hyper parameters
        self.init_m = init_m
        self.ldk = ldk
        self.dhs_model_prediction_loss = nn.MSELoss()

    def train(self, dataloder):
        statistics = defaultdict(list)
        for (x, y) in dataloder:
            # energy head training
            neg_x = sample_langevin(x, self.surrogate_model, self.init_m, self.ldk, noise=True) 
            
            pos_energy = self.dhs_model(x)
            neg_energy = self.dhs_model(neg_x)
            energy_loss = pos_energy.mean() - neg_energy.mean()
            energy_loss += torch.pow(pos_energy, 2).mean() + torch.pow(neg_energy, 2).mean()

            energy_loss = energy_loss.mean()
            self.dhs_model_energy_opt.zero_grad()
            energy_loss.backward()
            self.dhs_model_energy_opt.step()

            statistics[f'train/energy_cdloss'].append(energy_loss)
        

        return statistics

    def launch(self, train_dl, epochs, retrain_model=True):
        if not retrain_model and os.path.exists(self.dhs_model.save_path):
            self.dhs_model.load()
            return 
        for e in tqdm(range(epochs), desc="train epochs"):
            self.train(train_dl)
        self.dhs_model.save()


class DualHeadSurogateModel(nn.Module):
    def __init__(self, input_size, hidden_size, which_obj, 
                 save_dir=None, save_prefix=None):
        super(DualHeadSurogateModel, self).__init__()
        self.n_dim = input_size
        self.n_obj = 1
        self.which_obj = which_obj
        activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]
        self.activate_functions = activate_functions

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size)-1], 1))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size
        
        self.save_path = os.path.join(save_dir, f"{save_prefix}-{which_obj}.pt")
    
    def forward(self, x):
        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            # print("x:",x)
            # print("x.shape:",x.shape)
            x = self.activate_functions[i](x)
        
        x = self.layers[len(self.hidden_size)](x)
        out = x

        return out
    
        
    def save(self, val_mse=None, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
            
        from offline_moo.off_moo_baselines.data import tkwargs
        
        self = self.to('cpu')
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_mse is not None:
            checkpoint["valid_mse"] = val_mse
        
        torch.save(checkpoint, save_path)
        self = self.to(**tkwargs)
    
    def load(self, save_path=None):
        assert self.save_path is not None or save_path is not None, "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        
        checkpoint = torch.load(save_path)
        self.load_state_dict(checkpoint["model_state_dict"])
        if 'valid_mse' in checkpoint.keys():
            valid_mse = checkpoint["valid_mse"]
            print(f"Successfully load trained model from {save_path} " 
                    f"with valid MSE = {valid_mse}")
        else:
            print(f"Successfully load trained model from {save_path} ")