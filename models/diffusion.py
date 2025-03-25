from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.unet import UNet


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t).float()
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        inference_time_step,
        unet,
        sigma_min,
        sigma_max,
        sigma_data,
        rho,
        S_churn,
        S_min,
        S_max,
        S_noise
    ):
        super().__init__()

        # member variables
        self.denoise_fn = UNet(**unet)
        self.inference_time_step = inference_time_step
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise

        # parameters
        sigma_min = max(self.S_min, self.sigma_min)
        sigma_max = min(self.S_max, self.sigma_max)
        
        # EDM style
        step_indices = torch.arange(self.inference_time_step)
        t_steps = torch.as_tensor(sigma_max ** (1 / self.rho) + step_indices / (self.inference_time_step - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        self.register_buffer('inference_time_steps', t_steps)
        
        # loss function
        self.loss_fn = partial(F.mse_loss, reduction="none")

    def inference(self, x_t, x_cond=None):
        ret = []
        
        # Main sampling loop.
        x_next = x_t * self.inference_time_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(self.inference_time_steps[:-1], self.inference_time_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(self.S_churn / self.inference_time_step, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)
            
            # No stochasticity.
            # t_hat = t_cur
            # x_hat = x_cur

            # Euler step.
            denoised = self.forward(x_hat, t_hat.flatten().reshape(-1, 1, 1, 1), x_cond)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.inference_time_step - 1:
                denoised = self.forward(x_next, t_next.flatten().reshape(-1, 1, 1, 1), x_cond)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            ret.append(x_next)
            
        return ret    

    @torch.no_grad()
    def fix_forward(self, x_0, time_steps=10, x_cond=None):
        sigma_min = max(self.S_min, self.sigma_min)
        sigma_max = min(self.S_max, self.sigma_max)
        step_indices = torch.arange(time_steps)
        t_steps = torch.as_tensor(sigma_max ** (1 / self.rho) + step_indices / (time_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).to(x_0.device) # t_N = 0
        
        x_next = x_0
        for i, (t_cur, t_next) in enumerate((zip(reversed(t_steps[:-1]), reversed(t_steps[1:])))):
            x_hat = x_next
            t_hat = t_next
            
            # Euler step.
            denoised = self.forward(x_hat, t_hat.flatten().reshape(-1, 1, 1, 1), x_cond)
            
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_cur - t_hat) * d_cur

            # Apply 2nd order correction.
            if i != 0:
                denoised = self.forward(x_next, t_cur.flatten().reshape(-1, 1, 1, 1), x_cond)
                d_prime = (x_next - denoised) / t_cur
                x_next = x_hat + (t_cur - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next

    def forward(self, x, sigma, x_cond):
        """
        :param[in]  x       torch.Tensor    [batch_size x channel x height x width]
        :param[in]  sigma       torch.Tensor    [batch_size x 1 x 1 x 1]
        :param[in]  x_cond  torch.Tensor    [batch_size x _ x height x width]
        """
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        F_x = self.denoise_fn(torch.cat([c_in * x, x_cond], dim=1), c_noise.flatten())
        D_x = c_skip * x + c_out * F_x
        
        return D_x
    
    def compute_loss(self, x_ref, D_x, weight):
        """
        :param[in]  x_ref       torch.Tensor    [batch_size x channel x height x width] 
        :param[in]  D_x       torch.Tensor    [batch_size x channel x height x weight]
        :param[in]  weight       torch.Tensor    [batch_size x 1 x 1 x 1]
        """
        
        # Ablation Study: Without loss-weights
        # return (weight * self.loss_fn(D_x, x_ref)).mean() 
        return self.loss_fn(D_x, x_ref).mean()