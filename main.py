import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from AnimeDiffusion import AnimeDiffusion
from distutils.util import strtobool
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration Parser')
    
    # Training Configuration
    parser.add_argument('--do_train', type=lambda x: bool(strtobool(x)), default=True, 
                    help='Enable or disable training (True/False)')
    parser.add_argument('--do_finetuning', type=lambda x: bool(strtobool(x)), default=False, 
                    help='Enable or disable finetuning (True/False)')
    parser.add_argument('--do_test', type=lambda x: bool(strtobool(x)), default=False, 
                        help='Enable or disable test (True/False)')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='Number of training epochs')
    parser.add_argument('--test_output_dir', type=str, 
                        default='./result_1/', 
                        help='Directory for test outputs')
    
    # Diffusion Process Configuration
    parser.add_argument('--inference_time_step', type=int, default=50, help='Number of diffusion inference time steps')
    
    # UNet Configuration
    parser.add_argument('--channel_in', type=int, default=7, 
                        help='Input channels for UNet')
    parser.add_argument('--channel_out', type=int, default=3, 
                        help='Output channels for UNet')
    parser.add_argument('--channel_mult', nargs='+', type=int, 
                        default=[1, 2, 4, 8], 
                        help='Channel multipliers for UNet')
    parser.add_argument('--attention_head', type=int, default=4, 
                        help='Number of attention heads')
    parser.add_argument('--cbam', type=bool, default=False, 
                        help='Enable or disable CBAM')
    
    # Optimizer Configuration
    parser.add_argument('--finetuning_loss_type', type=str, 
                        default='lpips', #mse, lpips, mse+lpips, mse+lpips+PCGrad
                        help='Path to reference data')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8, 
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=1, 
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay value')
    
    # Data Paths
    parser.add_argument('--train_reference_path', type=str, 
                        default='/data/Anime/train_data/reference/', 
                        help='Path to reference data')
    parser.add_argument('--train_condition_path', type=str, 
                        default='/data/Anime/train_data/sketch/', 
                        help='Path to condition data')
    parser.add_argument('--test_reference_path', type=str, 
                        default='/data/Anime/test_data/reference/', 
                        help='Path to reference data')
    parser.add_argument('--test_condition_path', type=str, 
                        default='/data/Anime/test_data/sketch/', 
                        help='Path to condition data')
    
    # Batch Sizes
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, 
                        help='Batch size for validation')
    
    # Image Size
    parser.add_argument('--size', type=int, default=256, 
                        help='Image size')
    
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    
    # EDM hyperparameters
    parser.add_argument('--sigma_min', type=float, default=0.002, help='Minimum noise level in the diffusion process')
    parser.add_argument('--sigma_max', type=float, default=80, help='Maximum noise level in the diffusion process')
    parser.add_argument('--sigma_data', type=float, default=0.5, help='Standard deviation of data distribution for normalizing noise levels')
    parser.add_argument('--rho', type=float, default=7, help='Controls the shape of the noise schedule (higher values lead to more aggressive noise scaling)')
    parser.add_argument('--P_mean', type=float, default=-1.2, help='Mean of the prior distribution used for sampling initial noise')
    parser.add_argument('--P_std', type=float, default=1.2, help='Standard deviation of the prior distribution for initial noise sampling')
    parser.add_argument('--S_churn', type=float, default=0, help='Probability of adding extra noise during sampling to improve diversity')
    parser.add_argument('--S_min', type=float, default=0, help='Minimum noise level where stochasticity is applied')
    parser.add_argument('--S_max', type=float, default=1000, help='Maximum noise level where stochasticity is applied')
    parser.add_argument('--S_noise', type=float, default=1, help='Scale factor for additional noise applied in stochastic sampling')
    parser.add_argument('--sampling_method_for_train_phase', type=str, default="log-uniform-importance", help='sampling method for train phase [log-normal, log-uniform, log-uniform-importance]')
    parser.add_argument(
    '--log_uniform_main_prob', type=float, default=0.9,
    help='Probability of sampling from the important region (e.g., log(σ) ∈ [-2, 1]) in log-uniform-importance sampling'
    )


    # Parse arguments
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    # Parse arguments
    cfg = parse_arguments()

    # Set seed for reproducibility
    pl.seed_everything(42)

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="train_avg_loss",
        filename='{epoch:02d}-{train_avg_loss:.4f}',
        save_top_k=3,
        mode="min",
        every_n_epochs=1,
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger("logs")

    # Trainer configuration
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.epochs,  # Use cfg.epochs instead of cfg.max_epochs
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true",
    )

    model = AnimeDiffusion(cfg)

    # Training
    if cfg.do_train:
        if cfg.do_finetuning:
            checkpoint = torch.load("/root/edm_AnimeDiffusion/logs/lightning_logs/version_0/checkpoints/epoch=03-train_avg_loss=0.1139.ckpt", map_location='cuda')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            trainer.fit(model)
        else: 
            trainer.fit(model)
    

    # Testing
    if cfg.do_test:
        os.makedirs(cfg.test_output_dir, exist_ok=True)  # 디렉토리 생성
        checkpoint = torch.load("/root/edm_AnimeDiffusion/logs/lightning_logs/version_5/checkpoints/epoch=00-train_avg_loss=0.0216.ckpt", map_location='cuda')
        checkpoint['state_dict'].pop('model.inference_time_steps', None)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        trainer.test(model)