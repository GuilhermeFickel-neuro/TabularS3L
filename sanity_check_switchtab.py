#!/usr/bin/env python3
"""
Sanity check: Can both SwitchTab models overfit on a tiny random dataset?
This script uses the training setup from train_switchtab.py to ensure consistency.
"""

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Model and config imports from the local project
from switchtab_matryoshka import (
    create_paper_exact_transformer_config,
)

# Imports from the training script and ts3l library
# We reuse the Lightning wrappers and configs to keep the sanity check consistent
# with the main training script.
from train_switchtab import (
    PaperExactSwitchTabLightning, 
    PaperExactSwitchTabMatryoshkaLightning
)
from ts3l.utils.embedding_utils import FTEmbeddingConfig
from ts3l.utils.switchtab_utils import (
    SwitchTabConfig, 
    SwitchTabDataset,
    SwitchTabFirstPhaseCollateFN,
)
from ts3l.utils.datamodule import TS3LDataModule


def create_tiny_random_dataset(n_samples=64, n_features=16, n_classes=4):
    """
    Create a tiny random dataset for the overfitting test.
    Returns pandas objects as expected by SwitchTabDataset.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # SwitchTabDataset expects pandas DataFrame for features and Series for labels
    features = {f'feat_{i}': np.random.randn(n_samples) for i in range(n_features)}
    X_df = pd.DataFrame(features)
    y_s = pd.Series(np.random.randint(0, n_classes, size=n_samples))
    
    continuous_cols = list(X_df.columns)
    category_cols = []
    
    print(f"Created dataset with {n_samples} samples, {n_features} features, and {n_classes} classes.")
    return X_df, y_s, continuous_cols, category_cols


class TinyDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for our tiny random dataset."""
    def __init__(self, config: SwitchTabConfig, batch_size: int):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.X, self.y, self.continuous_cols, self.category_cols = create_tiny_random_dataset(
            n_samples=64,
            n_features=config.embedding_config.input_dim,
            n_classes=config.output_dim
        )

    def setup(self, stage: str = None):
        # For the overfitting check, we'll use the supervised (second phase) setup.
        # This allows us to directly monitor classification accuracy.
        self.train_dataset = SwitchTabDataset(
            self.X, self.y.values, self.config, 
            continuous_cols=self.continuous_cols, 
            category_cols=self.category_cols, 
            is_second_phase=True
        )
        # Use the same data for validation to check if the model can memorize it.
        self.val_dataset = SwitchTabDataset(
            self.X, self.y.values, self.config,
            continuous_cols=self.continuous_cols,
            category_cols=self.category_cols,
            is_second_phase=True
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)


def create_sanity_check_config(
    n_features: int, n_classes: int, d_model: int, batch_size: int, max_epochs: int
) -> SwitchTabConfig:
    """Creates a SwitchTabConfig tailored for a quick overfitting sanity check."""
    
    embedding_config = FTEmbeddingConfig(
        input_dim=n_features,
        emb_dim=d_model,
        cont_nums=n_features,
        cat_cardinality=[],
        required_token_dim=2
    )
    
    backbone_config = create_paper_exact_transformer_config(d_model=d_model)
    
    # Calculate steps_per_epoch for the OneCycleLR scheduler
    n_samples = 64
    steps_per_epoch = (n_samples // batch_size) + (1 if n_samples % batch_size != 0 else 0)
    
    config = SwitchTabConfig(
        task="classification",
        embedding_config=embedding_config,
        backbone_config=backbone_config,
        output_dim=n_classes,
        corruption_rate=0.0, # Standard corruption rate
        loss_fn="CrossEntropyLoss",
        metric="accuracy",
        metric_hparams={'task': 'multiclass', 'num_classes': n_classes},
        optim="AdamW",
        optim_hparams={'lr': 1e-3},  # High learning rate for fast overfitting
        scheduler="OneCycleLR",
        scheduler_hparams={
            'max_lr': 1e-3, 
            'epochs': max_epochs, 
            'steps_per_epoch': steps_per_epoch,
            'pct_start': 0.3,
            'anneal_strategy': 'cos'
        }
    )
    return config


class OverfitCheckCallback(pl.Callback):
    """A PyTorch Lightning callback to stop training once high accuracy is achieved."""
    def __init__(self, monitor="val_accuracy", threshold=0.95):
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.overfit_achieved = False

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if self.monitor in logs:
            current_metric = logs[self.monitor].item()
            pl_module.log("overfit_check_metric", current_metric, prog_bar=True)
            if current_metric >= self.threshold:
                self.overfit_achieved = True
                trainer.should_stop = True
                print(f"\n✅ Overfitting successful! {self.monitor} reached {current_metric:.3f}")


def test_model_overfitting(
    model_class, config, datamodule, model_name, max_epochs, **kwargs
) -> bool:
    """
    Tests if a model can overfit on the tiny dataset using the PyTorch Lightning trainer.
    Now runs both pre-training and fine-tuning phases.
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    pl_model = model_class(config, **kwargs)

    # --- Phase 1: Pre-training ---
    print("\n--- Running Phase 1: Pre-training ---")
    pl_model.set_first_phase()

    train_ds_p1 = SwitchTabDataset(
        datamodule.X, datamodule.y.values, config, 
        continuous_cols=datamodule.continuous_cols, 
        category_cols=datamodule.category_cols, 
        is_second_phase=False
    )
    val_ds_p1 = SwitchTabDataset(
        datamodule.X, datamodule.y.values, config,
        continuous_cols=datamodule.continuous_cols,
        category_cols=datamodule.category_cols,
        is_second_phase=False
    )
    train_loader_p1 = DataLoader(
        train_ds_p1,
        batch_size=datamodule.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=SwitchTabFirstPhaseCollateFN()
    )
    val_loader_p1 = DataLoader(
        val_ds_p1,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=SwitchTabFirstPhaseCollateFN()
    )

    trainer_p1 = pl.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=max_epochs // 2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    try:
        trainer_p1.fit(pl_model, train_dataloaders=train_loader_p1, val_dataloaders=val_loader_p1)
    except Exception as e:
        import logging
        logging.error(f"Error during phase 1 training for {model_name}: {e}", exc_info=True)
        return False
    
    # --- Phase 2: Fine-tuning and Overfitting Check ---
    print("\n--- Running Phase 2: Fine-tuning & Overfitting Check ---")
    pl_model.set_second_phase(freeze_encoder=False)
    
    overfit_callback = OverfitCheckCallback(threshold=0.95)
    
    trainer_p2 = pl.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=max_epochs,
        callbacks=[overfit_callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    try:
        trainer_p2.fit(pl_model, datamodule=datamodule)
    except Exception as e:
        import logging
        logging.error(f"Error during phase 2 training for {model_name}: {e}", exc_info=True)
        return False

    if not overfit_callback.overfit_achieved:
         final_acc = trainer_p2.callback_metrics.get(overfit_callback.monitor, torch.tensor(0.0)).item()
         print(f"❌ {model_name} failed to overfit (final acc: {final_acc:.3f})")

    return overfit_callback.overfit_achieved


def main():
    """Main function to run the sanity checks."""
    print("SwitchTab Sanity Check: Can models overfit on tiny random data?")
    print("Using training configuration from train_switchtab.py for consistency.")
    print("="*70)
    
    # Common settings for the sanity check
    N_FEATURES = 16
    N_CLASSES = 4
    D_MODEL = 128
    BATCH_SIZE = 16
    MAX_EPOCHS = 1000 # Give it enough epochs to overfit

    config = create_sanity_check_config(
        n_features=N_FEATURES, 
        n_classes=N_CLASSES,
        d_model=D_MODEL,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS
    )
    
    datamodule = TinyDataModule(config, batch_size=BATCH_SIZE)

    # Test Regular SwitchTab
    regular_success = test_model_overfitting(
        PaperExactSwitchTabLightning,
        config,
        datamodule,
        "PaperExactSwitchTab",
        max_epochs=MAX_EPOCHS,
        temperature=-1.0 # Disable temperature scaling
    )
    
    # Test Matryoshka SwitchTab
    nesting_list = sorted(list(set([D_MODEL // 4, D_MODEL // 2, D_MODEL])))
    
    matryoshka_success = test_model_overfitting(
        PaperExactSwitchTabMatryoshkaLightning,
        config,
        datamodule,
        "PaperExactSwitchTabMatryoshka",
        max_epochs=MAX_EPOCHS,
        nesting_list=nesting_list,
        temperature=-1.0 # Disable temperature scaling
    )
    
    # Final Summary
    print(f"\n{'='*70}")
    print("SANITY CHECK RESULTS:")
    print(f"Regular SwitchTab:    {'✅ PASS' if regular_success else '❌ FAIL'}")
    print(f"Matryoshka SwitchTab: {'✅ PASS' if matryoshka_success else '❌ FAIL'}")
    print(f"{'='*70}")
    
    if regular_success and matryoshka_success:
        print("🎉 Both models can overfit! The basic implementations are likely correct.")
    elif regular_success and not matryoshka_success:
        print("🔍 Matryoshka model has issues. Check its specific implementation or loss function.")
    elif not regular_success and matryoshka_success:
        print("🤔 Unexpected: Matryoshka works but the regular model fails. Check the base model.")
    else:
        print("🚨 Both models failed to overfit. There might be a fundamental issue in the shared components.")

if __name__ == "__main__":
    main()
