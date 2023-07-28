from typing import Any, Callable, Dict, Optional, Tuple

import os
import sys
import json

import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
import wandb


class ToyDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.features = x
        self.labels = y
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]
    

class ANN(pl.LightningModule):
    def __init__(self, n_inputs, configs):
        super(ANN, self).__init__()
        
        #$ Set seed
        if configs['seed'] is not None:
            pl.seed_everything(configs['seed'])
        
        #$ Organize hidden layers
        hiddens_list = configs['network_architecture']
        hidden_layers = []
        for i, n in enumerate(hiddens_list[0:-1]):
            np1 = hiddens_list[i+1]
            hidden_layers.append(nn.Linear(n, np1))
            hidden_layers.append(nn.ReLU())
        hidden_layers = nn.ModuleList(hidden_layers)
        
        #$ Number of outputs
        n_outputs = 2
        
        #$ Create stack
        self.stack = nn.Sequential(
            nn.Linear(n_inputs, hiddens_list[0]),      # Input layer
            nn.ReLU(),
            *hidden_layers,                            # Hidden layers
            nn.Linear(hiddens_list[-1], n_outputs)     # Output layer
        )
        
        #$ Hyperparameter configs
        self.configs = configs
        
        #$ Initialize `switched` flag based on config['std_switch']
        if configs['std_switch']:
            self.freeze_std = True
        else:
            self.freeze_std = False
        
        #$ Loss function
        Loss = getattr(nn, configs['loss_func'])
        self.loss_fn = Loss()
        
        #$ Save hyperparameters
        self.save_hyperparameters()
        
        
    def forward(self, x):
        return self.stack(x)
    
    def training_step(self, batch, batch_idx):
        if isinstance(self.loss_fn, nn.GaussianNLLLoss):
            X, y = batch
            pred = self.stack(X.float())
            if self.freeze_std:
                loss = self.loss_fn(y.float().squeeze(), pred[:, 0], torch.tensor(1.))
            else:
                loss = self.loss_fn(y.float().squeeze(), pred[:,0], torch.exp(pred[:,1]))
                # loss = self.loss_fn(y.float(), 0., torch.exp(pred[:,1]))
            self.log('train/GaussianNLL', loss, on_step=False, on_epoch=True)
            
            #$ Also include MSE loss
            mse = F.mse_loss(y.float().squeeze(), pred[:,0])
            self.log('train/MSE', mse, on_step=False, on_epoch=True)
        else:
            raise NotImplementedError("Loss func not implemented")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if isinstance(self.loss_fn, nn.GaussianNLLLoss):
            X, y = batch
            pred = self.stack(X.float())
            if self.freeze_std:
                loss = self.loss_fn(y.float().squeeze(), pred[:, 0], torch.tensor(1.))
            else:
                loss = self.loss_fn(y.float().squeeze(), pred[:,0], torch.exp(pred[:,1]))
            self.log('val/GaussianNLL', loss, on_step=False, on_epoch=True)
            
            #$ Also include MSE loss
            mse = F.mse_loss(y.float().squeeze(), pred[:,0])
            self.log('val/MSE', mse, on_step=False, on_epoch=True)
        else:
            raise NotImplementedError("Loss func not implemented")
        
        return loss
    
    def configure_optimizers(self):
        optimizer_func = getattr(torch.optim, self.configs['optimizer'])
        if self.configs['l2'] is None:
            weight_decay = 0
        else:
            weight_decay = self.configs['l2']
        optimizer = optimizer_func(
            self.stack.parameters(),
            lr=self.configs['lr'],
            weight_decay=weight_decay
        )
        return optimizer

class LogLikelihoodSwitch(Callback):
    # Based on the EarlyStopping callback code from Pytorch Lightning,
    # this code has been modified to provide a switch condition for
    # the log likelihood loss.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}
    
    def __init__(
        self,
        monitor: str,
        min_delta: float=0.0,
        patience: int=3,
        mode: str="min",
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.wait_count = 0
        self.switched_epoch = 0
        
        if self.mode not in self.mode_dict:
            raise ValueError(
                f"`mode` can be {', '.join(self.mode_dict.keys())}, "
                "got {self.mode}"
            )

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)
    
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._check_on_train_epoch_end = (
            trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1
        )
        
    def _validate_condition_metric(self, logs: Dict[str, torch.Tensor]) -> bool:
        monitor_val = logs.get(self.monitor)
        
        error_msg = (
            f"LogLikelihoodSwitch conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `LogLikelihoodSwitch` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )
        
        if monitor_val is None:
            return False
        return True
    
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "switched_epoch": self.switched_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.switched_epoch = state_dict["switched_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end:
            return
        self._run_switching_check(trainer, pl_module)
        
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end:
            return
        self._run_switching_check(trainer, pl_module)
        
    def _run_switching_check(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logs = trainer.callback_metrics
        
        if not self._validate_condition_metric(logs):
            return
        
        current = logs[self.monitor].squeeze()
        should_switch, reason = self._evaluate_switching_criteria(current)
        should_switch = trainer.strategy.reduce_boolean_decision(should_switch, all=False)
        if should_switch:
            self.switched_epoch = trainer.current_epoch
            print(f"Switching training to train on mu and sigma (epoch {self.switched_epoch}")
            pl_module.freeze_std = False
    
    def _evaluate_switching_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_switch = False
        reason = None
        
        if self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_switch = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_switch = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to unfreeze std predictions."
                )
        
        return should_switch, reason
    
    def _improvement_message(self, current: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg