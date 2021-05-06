import itertools
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import Metric
from networks import Encoder, Generator, CriticX, CriticZ
from timeseries_anomalies import score_anomalies, find_anomalies
from utils import plot, plot_table_anomalies, plot_ts, unroll_ts, plot_rws
from contextual import contextual_accuracy, contextual_precision, contextual_recall, contextual_f1_score, contextual_prepare_weighted

def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.Linear, nn.Conv1d]:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class TadGAN(pl.LightningModule):
    def __init__(self, in_size:int, ts_size:int = 100, latent_dim: int = 20, lr:float = 0.0005, weight_decay:float = 1e-6,
                 iterations_critic:int = 5, gamma:float = 10, weighted: bool = True, use_gru=False):
        super(TadGAN, self).__init__()
        self.in_size = in_size
        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.iterations_critic = iterations_critic
        self.gamma = gamma
        self.weighted = weighted

        self.hparams = {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'iterations_critic': self.iterations_critic,
            'gamma': self.gamma
        }

        self.encoder = Encoder(in_size, ts_size=ts_size, out_size=self.latent_dim, batch_first=True, use_gru=use_gru)
        self.generator = Generator(use_gru=use_gru)
        self.critic_x = CriticX(in_size=in_size)
        self.critic_z = CriticZ()
        
        self.encoder.apply(init_weights)
        self.generator.apply(init_weights)
        self.critic_x.apply(init_weights)
        self.critic_z.apply(init_weights)
        
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
        
        self.y_hat = []
        self.index = []
        self.critic = []
        
    def on_fit_start(self):
        if self.logger is not None:
            fig = plot_rws(self.trainer.datamodule.X.cpu().numpy())
            self.logger.experiment.add_figure('Rolling windows/GT', fig, global_step=self.global_step)
        
    def forward(self, x):
        y_hat = self.generator(self.encoder(x))
        critic = self.critic_x(x)

        return y_hat, critic  

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch[0]
        batch_size = x.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        valid = -torch.ones(batch_size, 1, device=self.device)
        fake = torch.ones(batch_size, 1, device=self.device)

        if optimizer_idx == 0:
            if (batch_idx+1) % self.iterations_critic != 0:
                return None
            z_gen = self.encoder(x)
            x_rec = self.generator(z_gen)
            fake_gen_z = self.critic_z(z_gen)
            fake_gen_x = self.critic_x(self.generator(z))
            
            wx_loss = self._wasserstein_loss(valid, fake_gen_x)
            wz_loss = self._wasserstein_loss(valid, fake_gen_z)
            rec_loss = F.mse_loss(x_rec, x)
            loss = wx_loss + wz_loss + self.gamma * rec_loss          
            vals = {'train/Encoder_Generator/loss': loss, 'train/Encoder_Generator/Wasserstein_x_loss': wx_loss, 'train/Encoder_Generator/Wasserstein_z_loss': wz_loss, 'train/Encoder_Generator/Reconstruction_loss': rec_loss}
            self.log_dict(vals)      
        elif optimizer_idx == 1:
            valid_x = self.critic_x(x)
            x_gen = self.generator(z).detach()
            fake_x = self.critic_x(x_gen)
            
            wv_loss = self._wasserstein_loss(valid, valid_x)
            wf_loss = self._wasserstein_loss(fake, fake_x)
            gp_loss = self._calculate_gradient_penalty(self.critic_x, x, x_gen)
            loss = wv_loss + wf_loss + self.gamma * gp_loss
            vals = {'train/Critic_x/loss': loss, 'train/Critic_x/Wasserstein_valid_loss': wv_loss, 'train/Critic_x/Wasserstein_fake_loss': wf_loss, 'train/Critic_x/gradient_penalty': gp_loss}
            self.log_dict(vals)           
        elif optimizer_idx == 2:
            valid_z = self.critic_z(z)
            z_gen = self.encoder(x).detach()
            fake_z = self.critic_z(z_gen)
            
            wv_loss = self._wasserstein_loss(valid, valid_z)
            wf_loss = self._wasserstein_loss(fake, fake_z)
            gp_loss = self._calculate_gradient_penalty(self.critic_z, z, z_gen)
            loss = wv_loss + wf_loss + self.gamma * gp_loss
            vals = {'train/Critic_z/loss': loss, 'train/Critic_z/Wasserstein_valid_loss': wv_loss, 'train/Critic_z/Wasserstein_fake_loss': wf_loss, 'train/Critic_z/gradient_penalty': gp_loss}
            self.log_dict(vals)          
        else:
            raise NotImplementedError()
        return loss

    def validation_step(self, batch, batch_idx):
        x, index = batch
        y_hat, critic = self(x)

        self.y_hat.append(y_hat)
        self.index.append(index)
        self.critic.append(critic)
        return None
             
    def validation_epoch_end(self, validation_step_outputs):
        if self.logger is None:
            return
        
        for net_name, net in zip(['Encoder', 'Generator', 'Critic_X', 'Critic_Z'], [self.encoder, self.generator, self.critic_x, self.critic_z]):
            for m in net.modules():
                for name, param in m.named_parameters():
                    self.logger.experiment.add_histogram(net_name+'/'+name, param.data)
        
        y_hat = torch.cat(self.y_hat)
        critic = torch.cat(self.critic)
        index = torch.cat(self.index)
        
        self.index = []
        self.y_hat = []
        self.critic = []
          
        n_batches = self.all_gather(y_hat.size(0))
        max_n_batches = n_batches.max()
        if y_hat.size(0) < max_n_batches:
            diff = max_n_batches - y_hat.size(0)
            add_cols = torch.full((diff, *y_hat.shape[1:]), fill_value=float('nan'), dtype=y_hat.dtype, device=y_hat.device)
            y_hat = torch.cat((y_hat, add_cols))
            add_cols = torch.full((diff, *critic.shape[1:]), fill_value=float('nan'), dtype=critic.dtype, device=critic.device)
            critic = torch.cat((critic, add_cols))
            add_cols = torch.full((diff, *index.shape[1:]), fill_value=-1, dtype=index.dtype, device=index.device)
            index = torch.cat((index, add_cols))
        
        y_hat, critic, index = self.all_gather((y_hat, critic, index))
        
        if len(y_hat.shape) == 4:
            y_hat = torch.flatten(y_hat, 0, 1)
            critic = torch.flatten(critic, 0, 1)
            index = torch.flatten(index, 0, 1)
        dm = self.trainer.datamodule

        
        y_shape = y_hat.shape[1:]
        critic_shape = critic.shape[1:]
        index_shape = index.shape[1:]
        mask = ~torch.any(torch.flatten(y_hat, 1, -1).isnan(), dim=1)
        y_hat = y_hat[mask]
        y_hat = y_hat.view(y_hat.size(0), *y_shape)
        critic = critic[mask]
        critic = critic.view(critic.size(0), *critic_shape)
        index = index[mask]    
        
        idx = torch.argsort(index)        
        y_hat = y_hat[idx]
        critic = critic[idx]    
        
        assert y_hat.size(0) == critic.size(0)       
        
        max_idx = min(y_hat.shape[0], dm.X.shape[0])
        
        y_hat = y_hat[:max_idx].cpu().numpy()
        critic = critic = critic[:max_idx].cpu().numpy()
        X = dm.X[:max_idx].cpu().numpy()
        X_index = dm.X_index[:max_idx].cpu().numpy()
        index = dm.index
        
        self.y_hat = []
        self.critic = []

        # flatten the predicted windows
        # plot the time series
        fig = plot_ts([dm.y, unroll_ts(y_hat)], labels=['original', 'reconstructed'])
        self.logger.experiment.add_figure('TS reconstruction', fig, global_step=self.global_step)
        if y_hat.shape[0] == dm.X.shape[0]:
            fig = plot_rws(y_hat)
            self.logger.experiment.add_figure('Rolling windows/Reconstructed', fig, global_step=self.global_step)
        
        
        errors, true_index, true, predictions = score_anomalies(X,
                                                                y_hat,
                                                                critic,
                                                                X_index,
                                                                rec_error_type='dtw',
                                                                comb='mult')
        anomalies = find_anomalies(errors, index, window_size_portion=0.33,
                                   window_step_size_portion=0.1, fixed_threshold=True)
        if anomalies.size == 0:
            anomalies = pd.DataFrame(columns=['start', 'end', 'score'])
        else:
            anomalies = pd.DataFrame(anomalies, columns=['start', 'end', 'score'])
        
        gt_anomalies = dm.anomalies
        if gt_anomalies is not None:
            fig = plot(dm.df, [('anomalies', anomalies), ('gt_anomalies', gt_anomalies)])
        else:
            fig = plot(dm.df, [('anomalies', anomalies)])
        self.logger.experiment.add_figure('AD output', fig, global_step=self.global_step)
        
        metric_logged = False
        if not anomalies.empty:
            fig = plot_table_anomalies(anomalies)
            self.logger.experiment.add_figure('Anomaly table', fig, global_step=self.global_step)
              
            if gt_anomalies is not None:                
                # Workaround to dispay PR Curve
                if self.weighted:
                    labels, preds, weights = contextual_prepare_weighted(gt_anomalies, anomalies, data=dm.df)
                    self.logger.experiment.add_pr_curve('PR Curve', np.array(labels), np.array(preds), weights=np.array(weights), global_step=self.global_step)
                    
                acc = contextual_accuracy(gt_anomalies, anomalies, data=dm.df, weighted=self.weighted)
                prec = contextual_precision(gt_anomalies, anomalies, data=dm.df, weighted=self.weighted)
                recall = contextual_recall(gt_anomalies, anomalies, data=dm.df, weighted=self.weighted)
                f1 = contextual_f1_score(gt_anomalies, anomalies, data=dm.df, weighted=self.weighted, beta=2)
                vals = {'Accuracy': acc, 'Precision': prec, 'Recall': recall, 'F1': f1}
                self.log_dict(vals)
                metric_logged = True
        if not metric_logged:
            vals = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}
            self.log_dict(vals)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        params = [self.encoder.parameters(), self.generator.parameters()]
        e_g_opt = Adam(itertools.chain(*params), lr=self.lr, weight_decay=self.weight_decay)
        c_x_opt = Adam(self.critic_x.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        c_z_opt = Adam(self.critic_z.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return [e_g_opt, c_x_opt, c_z_opt]

    @staticmethod
    def _wasserstein_loss(y_true: torch.Tensor, y_pred: torch.Tensor):
        return torch.mean(y_true * y_pred)

    def _calculate_gradient_penalty(self, model: torch.nn.Module, y_true: torch.Tensor, y_pred: torch.Tensor):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake data
        alpha = torch.randn((y_true.size(0), 1, 1), device=self.device)
        # Get random interpolation between real and fake data
        interpolates = (alpha * y_true + ((1 - alpha) * y_pred)).requires_grad_(True)

        model_interpolates = model(interpolates)
        grad_outputs = torch.ones(model_interpolates.size(), device=self.device, requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=model_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

