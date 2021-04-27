import itertools
import math
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from networks import Encoder, Generator, CriticX, CriticZ
from timeseries_anomalies import score_anomalies, find_anomalies


class TadGAN(pl.LightningModule):
    def __init__(self, in_size:int, ts_size:int = 100, lr:float = 0.0005, weight_decay:float = 1e-6,
                 n_critic:int = 5, gamma:float = 10):
        super(TadGAN, self).__init__()
        self.in_size = in_size
        self.lr = lr
        self.weight_decay = weight_decay
        self._n_critic = n_critic
        self.gamma = gamma

        self.hparams = {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'n_critic': self._n_critic,
            'gamma': self.gamma
        }

        self.e = Encoder(in_size, ts_size=ts_size, batch_first=True)
        self.g = Generator()
        self.critic_x = CriticX()
        self.critic_z = CriticZ()

    def forward(self, x):
        y_hat = self.g(self.e(x))
        critic = self.critic_x(x)

        return y_hat, critic

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        batch_size = x.size(0)
        z = torch.randn(batch_size, self.in_size, device=self.device)
        valid = -torch.ones(batch_size, 1, device=self.device)
        fake = torch.ones(batch_size, 1, device=self.device)

        if optimizer_idx == 0:
            z_gen = self.e(x)
            x_rec = self.g(z_gen)
            fake_gen_z = self.critic_z(z_gen)
            fake_gen_x = self.critic_x(self.g(z))

            loss = self._wasserstein_loss(valid, fake_gen_x) + self._wasserstein_loss(valid, fake_gen_z) + \
                   self.gamma * F.mse_loss(x_rec, x)
            self.log('train_encoder_generator_loss', loss)
        elif optimizer_idx == 1:
            valid_x = self.critic_x(x)
            x_gen = self.g(z)
            fake_x = self.critic_x(x_gen)

            loss = self._wasserstein_loss(valid, valid_x) + self._wasserstein_loss(fake, fake_x) + \
                   self.gamma * self._calculate_gradient_penalty(self.critic_x, x, x_gen)
            self.log('train_critic_x_loss', loss)
        elif optimizer_idx == 2:
            valid_z = self.critic_z(z)
            z_gen = self.e(x)
            fake_z = self.critic_z(z_gen)

            loss = self._wasserstein_loss(valid, valid_z) + self._wasserstein_loss(fake, fake_z) + \
                   self.gamma * self._calculate_gradient_penalty(self.critic_z, z, z_gen)
            self.log('train_critic_z_loss', loss)
        else:
            raise NotImplementedError()
        return loss

    def validation_step(self, batch, batch_idx):
        x, index = batch
        index = index.cpu().numpy()
        y_hat, critic = self(x)
        errors, true_index, true, predictions = score_anomalies(x.cpu().numpy(),
                                                                y_hat.cpu().numpy(),
                                                                critic.cpu().numpy(),
                                                                index,
                                                                rec_error_type='dtw',
                                                                comb='sum',
                                                                lambda_rec=0.5)
        anomalies = find_anomalies(errors, index, window_size_portion=0.33,
                                   window_step_size_portion=0.1, fixed_threshold=True)
        self.print(anomalies)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        if optimizer_idx != 0 or (batch_idx + 1) % self._n_critic == 0:
            optimizer.step(closure=optimizer_closure)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        params = [self.e.parameters(), self.g.parameters()]
        e_g_opt = Adam(itertools.chain(*params), lr=self._lr, weight_decay=self.weight_decay)
        c_x_opt = Adam(self.critic_x.parameters(), lr=self._lr, weight_decay=self.weight_decay)
        c_z_opt = Adam(self.critic_z.parameters(), lr=self._lr, weight_decay=self.weight_decay)

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

