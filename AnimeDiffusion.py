import os
import time
import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.anime import Anime
from models.diffusion import GaussianDiffusion
import utils.image
import utils.path
import torch.nn.functional as F
import lpips
import torch.nn.utils as nn_utils

class AnimeDiffusion(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        self.save_hyperparameters(cfg)
        
        unet = {
            "channel_in": self.cfg.channel_in,
            "channel_out": self.cfg.channel_out,
            "channel_mult": self.cfg.channel_mult,
            "attention_head": self.cfg.attention_head,
            "cbam": self.cfg.cbam,
        }
        
        self.P_mean = self.cfg.P_mean
        self.P_std = self.cfg.P_std
        self.sigma_data = self.cfg.sigma_data
        
        self.model = GaussianDiffusion(
            inference_time_step=cfg.inference_time_step,
            unet=unet,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            sigma_data=cfg.sigma_data,
            rho=cfg.rho,
            S_churn=cfg.S_churn,
            S_min=cfg.S_min,
            S_max=cfg.S_max,
            S_noise=cfg.S_noise,
        )
        
        if "lpips" in cfg.finetuning_loss_type:
            self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.device) # LPIPS loss
            self.lpips_loss_fn.eval()
            self.lpips_loss_fn.requires_grad_(False)

        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr * len(self.cfg.gpus),
            weight_decay=self.cfg.weight_decay
        )

        def lr_lambda(step):
            epoch = (step / len(self.train_dataloader()) + self.current_epoch)
            
            # Warmup
            if epoch < self.cfg.warmup_epochs:
                warmup_ratio = epoch / self.cfg.warmup_epochs
                return warmup_ratio * (1 - self.cfg.min_lr/self.cfg.lr) + self.cfg.min_lr/self.cfg.lr

            # Cosine Decay
            progress = (epoch - self.cfg.warmup_epochs) / (self.cfg.epochs - self.cfg.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress)) * (1 - self.cfg.min_lr/self.cfg.lr) + self.cfg.min_lr/self.cfg.lr

        # Scheduler
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def train_dataloader(self):
        self.train_dataset = Anime(
            reference_path = self.cfg.train_reference_path, 
            condition_path = self.cfg.train_condition_path,
            size = self.cfg.size,
        )
        train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            pin_memory=True,
            drop_last=True
        )
        return train_dataloader

    def test_dataloader(self):
        self.test_dataset = Anime(
            reference_path = self.cfg.test_reference_path, 
            condition_path = self.cfg.test_condition_path,
            size = self.cfg.size,
        )
        test_dataset = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            pin_memory=True,
            drop_last=True
        )
        return test_dataset
    
    def on_train_start(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.print(f"Total Parameters: {total_params:,}")
        self.print(f"Trainable Parameters: {trainable_params:,}")

    def pretraining_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]
        
        # [B, 1, H, W] + [B, 3, H, W] → [B, 4, H, W]
        x_cond = torch.cat([x_con, x_dis], dim=1)
        
        if self.cfg.sampling_method_for_train_phase == "log-normal":
            rnd_normal = torch.randn([x_ref.shape[0], 1, 1, 1], device=self.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        elif self.cfg.sampling_method_for_train_phase == "log-uniform":
            rnd_uniform = torch.rand([x_ref.shape[0], 1, 1, 1], device=self.device)
            log_sigma_min = torch.log(torch.tensor(self.cfg.sigma_min, device=self.device))
            log_sigma_max = torch.log(torch.tensor(self.cfg.sigma_max, device=self.device))
            log_sigma = rnd_uniform * (log_sigma_max - log_sigma_min) + log_sigma_min
            sigma = log_sigma.exp()
        else:
            raise NotImplementedError(f"Unknown sampling method: {self.cfg.sampling_method_for_train_phase}")
        
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(x_ref) * sigma
        D_x = self.model(x_ref + noise, sigma, x_cond)
        loss = self.model.compute_loss(x_ref, D_x, weight)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, logger=True)
        
        return loss
    
    def finetuning_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]

        # [B, 1, H, W] + [B, 3, H, W] → [B, 4, H, W]
        x_cond = torch.cat([x_con, x_dis], dim=1)

        with torch.no_grad():
            x_T = self.model.fix_forward(x_ref, x_cond=x_cond)

        x_til = self.model.inference(x_t=x_T, x_cond=x_cond)[-1]
        
        if self.cfg.finetuning_loss_type == 'lpips':
            loss = self.lpips_loss_fn(x_til, x_ref).mean()
        elif self.cfg.finetuning_loss_type == 'mse':
            loss = F.mse_loss(x_til, x_ref)
        elif self.cfg.finetuning_loss_type == 'mse+lpips':
            mse_loss = F.mse_loss(x_til, x_ref)
            lpips_loss = self.lpips_loss_fn(x_til, x_ref).mean()
            self.log("mse_loss", mse_loss, prog_bar=True, sync_dist=True)
            self.log("lpips_loss", lpips_loss, prog_bar=True, sync_dist=True)
            loss = (mse_loss + lpips_loss) / 2
            
        else:
            raise ValueError(f"Invalid loss type: {self.cfg.finetuning_loss_type}")

        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        current_lr = optimizer.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, logger=True)
        
        return loss

    
    def training_step(self, batch, batch_idx):
        if self.cfg.do_finetuning:
            return self.finetuning_step(batch, batch_idx)
        else:
            return self.pretraining_step(batch, batch_idx)

    def on_train_epoch_end(self):
        avg_loss = self.all_gather(self.trainer.callback_metrics["train_loss"]).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)
        
        if self.cfg.do_finetuning and self.cfg.finetuning_loss_type == 'mse+lpips':
            avg_mse = self.all_gather(self.trainer.callback_metrics["mse_loss"]).mean()
            avg_lpips = self.all_gather(self.trainer.callback_metrics["lpips_loss"]).mean()
            self.log("avg_mse", avg_mse, prog_bar=True)
            self.log("avg_lpips", avg_lpips, prog_bar=True)
        
        if self.trainer.is_global_zero:
            self.print(f"Epoch {self.current_epoch} - Avg Loss: {avg_loss:.4f}")
            
    def test_step(self, batch, batch_idx):
        x_ref = batch["reference"].to(self.device)  # [B, 3, H, W]
        x_con = batch["condition"].to(self.device)  # [B, 1, H, W]
        x_dis = batch["distorted"].to(self.device)  # [B, 3, H, W]

        noise = torch.randn_like(x_ref).to(self.device)  # [B, 3, H, W]

        with torch.no_grad():
            rets = self.model.inference(
                x_t=noise,
                x_cond=torch.cat([x_con, x_dis], dim=1),
            )[-1]

        images = utils.image.tensor2PIL(rets)
        for i, filename in enumerate(batch['name']):
            output_path = os.path.join(self.cfg.test_output_dir, f'ret_{filename}')
            images[i].save(output_path)

    def on_test_epoch_end(self):
        self.print(f"All test outputs saved to {self.cfg.test_output_dir}")
