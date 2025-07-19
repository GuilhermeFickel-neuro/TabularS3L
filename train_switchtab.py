#!/usr/bin/env python3
"""
Simple training script for PaperExactSwitchTab and PaperExactSwitchTabMatryoshka
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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
from benchmark.datasets import load_diabetes


class PaperExactSwitchTabLightning(TS3LLightining):
    """Lightning wrapper for PaperExactSwitchTab"""
    def __init__(self, config):
        super().__init__(config)
        
    def _initialize(self, config):
        self.u_label = -1
        self.alpha = 1.0
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        self.model = PaperExactSwitchTab(
            embedding_config=config.embedding_config,
            backbone_config=config.backbone_config,
            output_dim=config.output_dim
        )

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
    def __init__(self, config, nesting_list=None):
        self.nesting_list = nesting_list
        super().__init__(config)
        
    def _initialize(self, config):
        self.u_label = -1
        self.alpha = 1.0
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        self.matryoshka_loss_fn = create_matryoshka_loss()
        self.model = PaperExactSwitchTabMatryoshka(
            embedding_config=config.embedding_config,
            backbone_config=config.backbone_config,
            output_dim=config.output_dim,
            nesting_list=self.nesting_list
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
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')],
        enable_progress_bar=False,
        enable_model_summary=False
    )
    trainer.fit(pl_model, datamodule=first_phase_datamodule)
    
    # Second phase training  
    pl_model.set_second_phase(freeze_encoder=False)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')],
        enable_progress_bar=False,
        enable_model_summary=False
    )
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


def main():
    print("Loading dataset...")
    data, label, continuous_cols, category_cols, output_dim, metric_name, metric_hparams = load_diabetes()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Dataset: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    print(f"Features: {len(continuous_cols)} continuous, {len(category_cols)} categorical")
    
    # Create configurations according to FT-transformer paper specifications
    # Using d_token=288 as specified in the original FT-transformer config
    d_token = 192
    
    embedding_config = FTEmbeddingConfig(
        input_dim=X_train.shape[1],
        emb_dim=d_token,  # d_token from FT-transformer config
        cont_nums=len(continuous_cols),
        cat_cardinality=get_category_cardinality(X_train, category_cols),
        required_token_dim=2  # Use 2 for transformer backbone (generates token sequence)
    )
    backbone_config = create_paper_exact_transformer_config(d_model=d_token)
    
    config = SwitchTabConfig(
        task="classification",
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=output_dim,
        loss_fn="CrossEntropyLoss",
        metric=metric_name,
        optim="Adam",
        optim_hparams={'lr': 0.001}
    )
    
    # Create datasets for first phase (pretraining)
    train_ds_phase1 = SwitchTabDataset(X_train, y_train.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=False)
    val_ds_phase1 = SwitchTabDataset(X_val, y_val.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=False)
    
    # Create datasets for second phase (fine-tuning)
    train_ds_phase2 = SwitchTabDataset(X_train, y_train.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
    val_ds_phase2 = SwitchTabDataset(X_val, y_val.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
    test_ds = SwitchTabDataset(X_test, y_test.values, config, continuous_cols=continuous_cols, category_cols=category_cols, is_second_phase=True)
    
    # Create dataloaders for first phase (with special collate function)
    first_phase_dl = TS3LDataModule(train_ds_phase1, val_ds_phase1, batch_size=32, train_sampler="random", 
                                    train_collate_fn=SwitchTabFirstPhaseCollateFN(), 
                                    valid_collate_fn=SwitchTabFirstPhaseCollateFN())
    
    # Create dataloaders for second phase (standard collate function)
    second_phase_dl = TS3LDataModule(train_ds_phase2, val_ds_phase2, batch_size=32, train_sampler="random")
    
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
    
    print("\n" + "="*60)
    print("Training PaperExactSwitchTab...")
    print("="*60)
    
    # Train paper-exact SwitchTab
    exact_model = train_model(PaperExactSwitchTabLightning, config, first_phase_dl, second_phase_dl)
    
    print("\n" + "="*60) 
    print("Training PaperExactSwitchTabMatryoshka...")
    print("="*60)
    
    # Train paper-exact SwitchTab with Matryoshka
    nesting_list = [X_train.shape[1]//4, X_train.shape[1]//2, 3*X_train.shape[1]//4, X_train.shape[1]]
    matryoshka_model = train_model(
        lambda config: PaperExactSwitchTabMatryoshkaLightning(config, nesting_list), 
        config, first_phase_dl, second_phase_dl
    )
    
    print("\n" + "="*60)
    print("Extracting embeddings...")
    print("="*60)
    
    # Extract embeddings
    exact_embeddings, exact_salient = extract_embeddings(exact_model, test_dl)
    matryoshka_embeddings, matryoshka_salient = extract_embeddings(matryoshka_model, test_dl)
    
    print(f"\nResults:")
    print(f"Paper-exact SwitchTab embeddings shape: {exact_embeddings.shape}")
    if exact_salient is not None:
        print(f"Paper-exact salient embeddings shape: {exact_salient.shape}")
    
    print(f"Matryoshka embeddings shape: {matryoshka_embeddings.shape}")
    if matryoshka_salient is not None:
        print(f"Matryoshka salient embeddings shape: {matryoshka_salient.shape}")
    
    print(f"Nesting dimensions: {nesting_list}")
    
    print("\n✓ Training completed successfully!")
    return exact_model, matryoshka_model, exact_embeddings, matryoshka_embeddings


if __name__ == "__main__":
    main() 