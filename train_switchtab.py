#!/usr/bin/env python3
"""
Simple training script for PaperExactSwitchTab and PaperExactSwitchTabMatryoshka
"""

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner import Tuner
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Import the paper-exact implementations
from switchtab_matryoshka import (
    PaperExactSwitchTab, PaperExactSwitchTabMatryoshka,
    create_paper_exact_transformer_config, create_matryoshka_loss
)

# Import existing utilities
from ts3l.utils.embedding_utils import FTEmbeddingConfig
from ts3l.utils.switchtab_utils import SwitchTabConfig, SwitchTabDataset, SwitchTabFirstPhaseCollateFN
from ts3l.utils import TS3LDataModule, get_category_cardinality
from ts3l.pl_modules.base_module import TS3LLightining

# Load simple dataset
from benchmark.datasets import load_higgs

# Import custom dataset loader
from custom_dataset import load_custom_dataset_with_split, load_custom_dataset


class PaperExactSwitchTabLightning(TS3LLightining):
    """Lightning wrapper for PaperExactSwitchTab"""
    def __init__(self, config, temperature=-1.0, use_transformer_projector=False, 
                 projector_n_heads=8, projector_dim_feedforward=2048, projector_dropout=0.1):
        self.temperature = temperature
        self.use_transformer_projector = use_transformer_projector
        self.projector_n_heads = projector_n_heads
        self.projector_dim_feedforward = projector_dim_feedforward
        self.projector_dropout = projector_dropout
        super().__init__(config)
        
    def _initialize(self, config):
        self.u_label = -1
        self.alpha = 1.0
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        self.model = PaperExactSwitchTab(
            embedding_config=config.embedding_config,
            backbone_config=config.backbone_config,
            output_dim=config.output_dim,
            temperature=self.temperature,
            use_transformer_projector=self.use_transformer_projector,
            projector_n_heads=self.projector_n_heads,
            projector_dim_feedforward=self.projector_dim_feedforward,
            projector_dropout=self.projector_dropout
        )

    def on_train_epoch_end(self):
        """Log current learning rate at the end of each epoch"""
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True)
        print('')

    def _get_first_phase_loss(self, batch):
        x_orig, x_corr, y = batch
        size = len(x_orig) // 2
        
        # Combine data for switching
        x_combined = torch.cat([x_corr[:size], x_corr[size:]])
        
        x_hat, y_hat = self.model._first_phase_step(x_combined)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss_fn(x_hat, torch.cat([x_orig[:size], x_orig[size:], x_orig[:size], x_orig[size:]]))
        
        # Task loss (only for labeled data)
        labeled_mask = y != self.u_label
        if labeled_mask.any():
            task_loss = self.task_loss_fn(y_hat[labeled_mask], y[labeled_mask])
        else:
            task_loss = torch.tensor(0.0)
            
        return recon_loss + self.alpha * task_loss

    def _get_second_phase_loss(self, batch):
        x, y = batch
        y_hat = self.model._second_phase_step(x)
        task_loss = self.task_loss_fn(y_hat, y)
        return task_loss, y, y_hat

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.model._second_phase_step(x)


class PaperExactSwitchTabMatryoshkaLightning(PaperExactSwitchTabLightning):
    """Lightning wrapper for PaperExactSwitchTabMatryoshka"""
    def __init__(self, config, nesting_list=None, temperature=-1.0, use_transformer_projector=False,
                 projector_n_heads=8, projector_dim_feedforward=2048, projector_dropout=0.1):
        self.nesting_list = nesting_list
        super().__init__(config, temperature=temperature, use_transformer_projector=use_transformer_projector,
                         projector_n_heads=projector_n_heads, projector_dim_feedforward=projector_dim_feedforward,
                         projector_dropout=projector_dropout)
        
    def _initialize(self, config):
        self.u_label = -1
        self.alpha = 1.0
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        self.matryoshka_loss_fn = create_matryoshka_loss()
        self.model = PaperExactSwitchTabMatryoshka(
            embedding_config=config.embedding_config,
            backbone_config=config.backbone_config,
            output_dim=config.output_dim,
            nesting_list=self.nesting_list,
            temperature=self.temperature,
            use_transformer_projector=self.use_transformer_projector,
            projector_n_heads=self.projector_n_heads,
            projector_dim_feedforward=self.projector_dim_feedforward,
            projector_dropout=self.projector_dropout
        )

    def _get_first_phase_loss(self, batch):
        x_orig, x_corr, y = batch
        size = len(x_orig) // 2
        
        x_combined = torch.cat([x_corr[:size], x_corr[size:]])
        x_hat, y_hat_nested = self.model._first_phase_step(x_combined)
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss_fn(x_hat, torch.cat([x_orig[:size], x_orig[size:], x_orig[:size], x_orig[size:]]))
        
        # Matryoshka task loss
        labeled_mask = y != self.u_label
        if labeled_mask.any():
            task_loss = self.matryoshka_loss_fn(y_hat_nested, y[labeled_mask])
        else:
            task_loss = torch.tensor(0.0)
            
        return recon_loss + self.alpha * task_loss

    def _get_second_phase_loss(self, batch):
        x, y = batch
        y_hat_nested = self.model._second_phase_step(x)
        task_loss = self.matryoshka_loss_fn(y_hat_nested, y)
        return task_loss, y, y_hat_nested[0]  # Use first nested output for metrics


def train_model(model_class, config, first_phase_datamodule, second_phase_datamodule, max_epochs=10):
    """Train a model with early stopping"""
    pl_model = model_class(config)
    
    # First phase training
    pl_model.set_first_phase()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=max_epochs//2, mode='min')],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    trainer.fit(pl_model, datamodule=first_phase_datamodule)
    
    # Second phase training  
    pl_model.set_second_phase(freeze_encoder=False)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=max_epochs//2, mode='min')],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    trainer.fit(pl_model, datamodule=second_phase_datamodule)
    
    return pl_model


def train_model_with_lr_finder(model_class, config, first_phase_datamodule, second_phase_datamodule, max_epochs=10, use_lr_finder=True):
    """Train a model with LR finder and learning rate scheduling"""
    pl_model = model_class(config)
    
    # First phase training
    pl_model.set_first_phase()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=max_epochs//2, mode='min')],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Optional LR finder for first phase
    if use_lr_finder:
        print("Running LR finder for first phase...")
        tuner = Tuner(trainer)
        # Temporarily add lr attribute for LR finder
        pl_model.lr = pl_model.optim_hparams['lr']
        lr_finder = tuner.lr_find(pl_model, datamodule=first_phase_datamodule, attr_name='lr')
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested LR for first phase: {suggested_lr}")
        
        # Update the learning rate in both places
        pl_model.optim_hparams['lr'] = suggested_lr
        pl_model.lr = suggested_lr
        
        # Update OneCycleLR max_lr to use the suggested learning rate
        if pl_model.scheduler_hparams and 'max_lr' in pl_model.scheduler_hparams:
            pl_model.scheduler_hparams['max_lr'] = suggested_lr
            print(f"Updated OneCycleLR max_lr to: {suggested_lr}")
        
        # Plot the LR finder results
        fig = lr_finder.plot(suggest=True)
        fig.show()
    
    trainer.fit(pl_model, datamodule=first_phase_datamodule)
    
    # Second phase training  
    pl_model.set_second_phase(freeze_encoder=False)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=max_epochs//2, mode='min')],
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Optional LR finder for second phase
    if use_lr_finder:
        print("Running LR finder for second phase...")
        tuner = Tuner(trainer)
        # Temporarily add lr attribute for LR finder
        pl_model.lr = pl_model.optim_hparams['lr']
        lr_finder = tuner.lr_find(pl_model, datamodule=second_phase_datamodule, attr_name='lr')
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested LR for second phase: {suggested_lr}")
        
        # Update the learning rate in both places
        pl_model.optim_hparams['lr'] = suggested_lr
        pl_model.lr = suggested_lr
        
        # Update OneCycleLR max_lr to use the suggested learning rate
        if pl_model.scheduler_hparams and 'max_lr' in pl_model.scheduler_hparams:
            pl_model.scheduler_hparams['max_lr'] = suggested_lr
            print(f"Updated OneCycleLR max_lr to: {suggested_lr}")
        
        # Plot the LR finder results
        fig = lr_finder.plot(suggest=True)
        fig.show()
    
    trainer.fit(pl_model, datamodule=second_phase_datamodule)
    
    return pl_model


def extract_embeddings(model, dataloader):
    """Extract embeddings from trained model"""
    model.eval()
    embeddings = []
    salient_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            # Set to return salient features
            model.model.return_salient_feature = True
            
            # Get embeddings
            if hasattr(model.model, '_second_phase_step'):
                output = model.model._second_phase_step(x)
                if isinstance(output, tuple):
                    _, salient = output
                    salient_embeddings.append(salient)
                
            # Get encoder embeddings
            x_emb = model.model.embedding_module(x)
            encoder_emb = model.model.encoder(x_emb)
            embeddings.append(encoder_emb)
    
    return torch.cat(embeddings), torch.cat(salient_embeddings) if salient_embeddings else None


def compute_ks_metric(model, dataloader):
    """Compute KS metric on test dataset using scipy.stats.ks_2samp"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            logits = model.predict_step(batch, 0)
            
            # Handle matryoshka output
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            
            # Convert to probabilities
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            else:
                probs = torch.sigmoid(logits.squeeze()).cpu()
                
            all_probs.append(probs)
            all_labels.append(y.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    
    ks_stat, _ = stats.ks_2samp(neg_probs, pos_probs)
    return ks_stat, pos_probs, neg_probs


def main(use_lr_finder=True):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SwitchTab models')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--d_token', type=int, default=512, help='Token dimension for transformer (default: 512)')
    parser.add_argument('--corruption_rate', type=float, default=0.3, help='Corruption rate for feature corruption (default: 0.3)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of dataloader workers (default: auto-detect based on CPU cores)')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='Prefetch factor for dataloader (default: 4)')
    parser.add_argument('--temperature', type=float, default=-1.0, help='Temperature for logit normalization, -1 disables it (default: -1.0)')
    parser.add_argument('--use_transformer_projector', action='store_true', help='Use TransformerEncoderLayer instead of _PaperProjector (default: False)')
    parser.add_argument('--projector_n_heads', type=int, default=8, help='Number of heads for transformer projector (default: 8)')
    parser.add_argument('--projector_dim_feedforward', type=int, default=2048, help='Feedforward dimension for transformer projector (default: 2048)')
    parser.add_argument('--projector_dropout', type=float, default=0.1, help='Dropout rate for transformer projector (default: 0.1)')
    
    # Custom dataset arguments
    parser.add_argument('--train_path', type=str, default=None, help='Path to custom training CSV file (tab-separated). If provided, uses custom dataset instead of Higgs.')
    parser.add_argument('--test_path', type=str, default=None, help='Path to custom test CSV file (tab-separated). Optional - if not provided, will split train_path.')
    parser.add_argument('--target', type=str, default=None, help='Name of the target column in custom dataset.')
    parser.add_argument('--max_size', type=int, default=100000, help='Maximum number of rows to load from custom dataset to prevent OOM (default: 100000)')
    
    args = parser.parse_args()
    
    # Auto-detect optimal number of workers if not specified
    if args.num_workers is None:
        import os
        # Use 75% of available CPU cores, but at least 4 and at most 16
        num_cores = os.cpu_count() or 4
        args.num_workers = max(4, min(16, int(num_cores * 0.75)))
        print(f"Auto-detected {args.num_workers} workers based on {num_cores} CPU cores")
    else:
        print(f"Using {args.num_workers} workers as specified")
    
    # Optimize for Tensor Cores on RTX GPUs
    torch.set_float32_matmul_precision('medium')
    
    # Check if custom dataset is specified
    if args.train_path:
        if not args.target:
            raise ValueError("--target must be specified when using custom dataset (--train_path)")
        
        print("Loading custom dataset...")
        if args.test_path:
            # Load train and test separately
            X_train_full, X_test, y_train_full, y_test, continuous_cols, category_cols, output_dim, metric_name, metric_hparams = load_custom_dataset(
                args.train_path, args.test_path, args.target, max_size=args.max_size
            )
            # Split train into train/val
            X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=True)
        else:
            # Load and split single file
            X_train, X_val, X_test, y_train, y_val, y_test, continuous_cols, category_cols, output_dim, metric_name, metric_hparams = load_custom_dataset_with_split(
                args.train_path, args.target, test_size=0.2, val_size=0.2, random_state=42, max_size=args.max_size
            )
    else:
        # Use default Higgs dataset
        print("Loading Higgs dataset...")
        data, label, continuous_cols, category_cols, output_dim, metric_name, metric_hparams = load_higgs()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    print(f"Features: {len(continuous_cols)} continuous, {len(category_cols)} categorical")
    
    # Create configurations according to FT-transformer paper specifications
    # Using d_token from command line argument
    d_token = args.d_token
    
    embedding_config = FTEmbeddingConfig(
        input_dim=X_train.shape[1],
        emb_dim=d_token,  # d_token from FT-transformer config
        cont_nums=len(continuous_cols),
        cat_cardinality=get_category_cardinality(X_train, category_cols),
        required_token_dim=2  # Use 2 for transformer backbone (generates token sequence)
    )
    backbone_config = create_paper_exact_transformer_config(d_model=d_token)
    
    # Calculate steps_per_epoch for OneCycleLR
    batch_size = args.batch_size
    max_epochs = args.epochs
    steps_per_epoch = len(X_train) // batch_size + (1 if len(X_train) % batch_size != 0 else 0)
    
    config = SwitchTabConfig(
        task="classification",
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=output_dim,
        corruption_rate=args.corruption_rate,
        loss_fn="CrossEntropyLoss",
        metric=metric_name,
        optim="RMSprop",
        optim_hparams={'lr': 0.0003},  # Initial LR, will be updated by LR finder
        scheduler="OneCycleLR",  # OneCycleLR for better convergence
        scheduler_hparams={'max_lr': 0.0003, 'epochs': max_epochs, 'steps_per_epoch': steps_per_epoch, 'pct_start': 0.3, 'anneal_strategy': 'cos'}  # OneCycleLR params
    )
    
    # Create datasets for first phase (pretraining)
    train_ds_phase1 = SwitchTabDataset(X_train, y_train.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=False)
    val_ds_phase1 = SwitchTabDataset(X_val, y_val.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=False)
    
    # Create datasets for second phase (fine-tuning)
    train_ds_phase2 = SwitchTabDataset(X_train, y_train.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
    val_ds_phase2 = SwitchTabDataset(X_val, y_val.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
    test_ds = SwitchTabDataset(X_test, y_test.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
    
    # Create optimized dataloaders for first phase (with special collate function)
    first_phase_dl = TS3LDataModule(train_ds_phase1, val_ds_phase1, 
                                    batch_size=batch_size, 
                                    n_jobs=args.num_workers, 
                                    train_sampler="random", 
                                    train_collate_fn=SwitchTabFirstPhaseCollateFN(), 
                                    valid_collate_fn=SwitchTabFirstPhaseCollateFN(),
                                    prefetch_factor=args.prefetch_factor,
                                    persistent_workers=True,
                                    pin_memory=True)
    
    # Create optimized dataloaders for second phase (standard collate function)
    second_phase_dl = TS3LDataModule(train_ds_phase2, val_ds_phase2, 
                                     batch_size=batch_size, 
                                     n_jobs=args.num_workers, 
                                     train_sampler="random",
                                     prefetch_factor=args.prefetch_factor,
                                     persistent_workers=True,
                                     pin_memory=True)
    
    test_dl = torch.utils.data.DataLoader(test_ds, 
                                          batch_size=batch_size, 
                                          shuffle=False, 
                                          num_workers=args.num_workers//2,  # Use fewer workers for test
                                          prefetch_factor=args.prefetch_factor,
                                          persistent_workers=True if args.num_workers > 0 else False,
                                          pin_memory=True)
    
    print("\n" + "="*60)
    print(f"Training PaperExactSwitchTab...")
    print(f"Projector type: {'TransformerEncoderLayer' if args.use_transformer_projector else '_PaperProjector'}")
    if args.use_transformer_projector:
        print(f"  - Heads: {args.projector_n_heads}")
        print(f"  - Feedforward dim: {args.projector_dim_feedforward}")
        print(f"  - Dropout: {args.projector_dropout}")
    print("="*60)
    
    # Train paper-exact SwitchTab with LR finder
    exact_model = train_model_with_lr_finder(
        lambda config: PaperExactSwitchTabLightning(
            config, 
            temperature=args.temperature,
            use_transformer_projector=args.use_transformer_projector,
            projector_n_heads=args.projector_n_heads,
            projector_dim_feedforward=args.projector_dim_feedforward,
            projector_dropout=args.projector_dropout
        ), 
        config, first_phase_dl, second_phase_dl, max_epochs=max_epochs, use_lr_finder=use_lr_finder
    )
    
    print("\n" + "="*60) 
    print("Training PaperExactSwitchTabMatryoshka...")
    print("="*60)
    
    # Train paper-exact SwitchTab with Matryoshka
    # Use encoder output dimension (d_token) for nesting, not input feature dimension
    encoder_dim = d_token  # This is the backbone output dimension
    nesting_list = [encoder_dim//4, encoder_dim//2, 3*encoder_dim//4, encoder_dim]
    matryoshka_model = train_model_with_lr_finder(
        lambda config: PaperExactSwitchTabMatryoshkaLightning(
            config, 
            nesting_list=nesting_list,
            temperature=args.temperature,
            use_transformer_projector=args.use_transformer_projector,
            projector_n_heads=args.projector_n_heads,
            projector_dim_feedforward=args.projector_dim_feedforward,
            projector_dropout=args.projector_dropout
        ), 
        config, first_phase_dl, second_phase_dl, max_epochs=max_epochs, use_lr_finder=use_lr_finder
    )
    
    print("\n" + "="*60)
    print("Computing KS metrics...")
    print("="*60)
    
    # Compute KS metrics
    exact_ks, exact_pos, exact_neg = compute_ks_metric(exact_model, test_dl)
    matryoshka_ks, matryoshka_pos, matryoshka_neg = compute_ks_metric(matryoshka_model, test_dl)
    
    print(f"\nKS Metrics:")
    print(f"Paper-exact SwitchTab KS: {exact_ks:.4f}")
    print(f"Matryoshka SwitchTab KS: {matryoshka_ks:.4f}")
    
    # Plot KS comparison
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Probability distributions
    plt.subplot(1, 2, 1)
    plt.hist(exact_pos, bins=50, alpha=0.7, label='Exact - Positive', density=True)
    plt.hist(exact_neg, bins=50, alpha=0.7, label='Exact - Negative', density=True)
    plt.hist(matryoshka_pos, bins=50, alpha=0.7, label='Matryoshka - Positive', density=True)
    plt.hist(matryoshka_neg, bins=50, alpha=0.7, label='Matryoshka - Negative', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distributions by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: KS comparison bar chart
    plt.subplot(1, 2, 2)
    models = ['PaperExact\nSwitchTab', 'Matryoshka\nSwitchTab']
    ks_values = [exact_ks, matryoshka_ks]
    bars = plt.bar(models, ks_values, color=['skyblue', 'lightcoral'])
    plt.ylabel('KS Statistic')
    plt.title('KS Metric Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, ks_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Training and evaluation completed successfully!")
    return exact_model, matryoshka_model, exact_ks, matryoshka_ks


if __name__ == "__main__":
    # Set use_lr_finder=False to disable automatic learning rate finding
    main(use_lr_finder=True) 