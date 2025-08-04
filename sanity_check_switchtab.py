#!/usr/bin/env python3
"""
Sanity check: Can both SwitchTab models overfit on a tiny random dataset?
"""

import torch
import torch.nn as nn
import numpy as np
from switchtab_matryoshka import (
    PaperExactSwitchTab, 
    PaperExactSwitchTabMatryoshka, 
    create_matryoshka_loss,
    create_paper_exact_transformer_config
)

def create_tiny_random_dataset(n_samples=32, n_features=10, n_classes=2):
    """Create a tiny random dataset for overfitting test"""
    torch.manual_seed(42)
    np.random.seed(42)
    
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    
    print(f"Created dataset: {X.shape}, classes: {y.unique().tolist()}")
    return X, y

def create_model_config(n_features, n_classes):
    """Create simple config for both models"""
    from ts3l.utils.embedding_utils import FTEmbeddingConfig
    
    # Use Feature Tokenizer embedding with proper dimensions
    embedding_config = FTEmbeddingConfig(
        input_dim=n_features,
        emb_dim=128,  # Match the transformer d_model
        cont_nums=n_features,
        cat_cardinality=[],
        required_token_dim=2 # Use 2 for transformer backbone
    )
    
    # Simple transformer config using the paper-exact function
    backbone_config = create_paper_exact_transformer_config(
        d_model=128,  # Small dimension for quick training
        is_embedded=True # Input is already embedded
    )
    
    class SimpleConfig:
        def __init__(self):
            self.embedding_config = embedding_config
            self.backbone_config = backbone_config
            self.output_dim = n_classes
    
    return SimpleConfig()

def test_model_overfitting(model, X, y, model_name, max_epochs=20):
    """Test if a model can overfit on the tiny dataset"""
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    # Simple loss and optimizer
    if "Matryoshka" in model_name:
        loss_fn = create_matryoshka_loss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # High LR for quick overfitting
    
    model.train()
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        try:
            if "Matryoshka" in model_name:
                # Matryoshka model returns tuple of outputs
                # The forward pass now correctly handles training mode
                x_hat, y_hat_nested = model(X)
                
                # We need a reconstruction loss for the first phase
                # Since this is a sanity check, we can use a dummy reconstruction target
                dummy_recon_target = torch.randn_like(x_hat)
                recon_loss = nn.MSELoss()(x_hat, dummy_recon_target)

                # Use the nested outputs for the task loss
                task_loss = loss_fn(y_hat_nested, y)
                loss = recon_loss + task_loss

                # Use the largest/last nested output for accuracy
                pred_logits = y_hat_nested[-1]
            else:
                # Regular model in training mode
                x_hat, y_hat = model(X)
                
                dummy_recon_target = torch.randn_like(x_hat)
                recon_loss = nn.MSELoss()(x_hat, dummy_recon_target)
                task_loss = loss_fn(y_hat, y)
                loss = recon_loss + task_loss

                pred_logits = y_hat
            
            loss.backward()
            optimizer.step()
            
            # Check accuracy
            with torch.no_grad():
                predictions = torch.argmax(pred_logits, dim=1)
                accuracy = (predictions == y).float().mean().item()
            
            print(f"Epoch {epoch:2d}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.3f}")
            
            # Early success check
            if accuracy > 0.95:
                print(f"✅ {model_name} successfully overfitted!")
                return True
                
        except Exception as e:
            import logging
            logging.error(f"Error in epoch {epoch}: {e}", exc_info=True)
            continue
    
    print(f"❌ {model_name} failed to overfit (final acc: {accuracy:.3f}, loss: {loss.item():.4f})")
    return False

def main():
    print("SwitchTab Sanity Check: Can models overfit on tiny random data?")
    print("="*70)
    
    # Create tiny dataset
    X, y = create_tiny_random_dataset(n_samples=32, n_features=10, n_classes=2)
    config = create_model_config(n_features=10, n_classes=2)
    
    # Test Regular SwitchTab
    try:
        regular_model = PaperExactSwitchTab(
            embedding_config=config.embedding_config,
            backbone_config=config.backbone_config,
            output_dim=config.output_dim,
            temperature=-1.0  # Disable temperature
        )
        regular_success = test_model_overfitting(regular_model, X, y, "PaperExactSwitchTab")
    except Exception as e:
        print(f"❌ Regular SwitchTab failed with error: {e}")
        regular_success = False
    
    # Test Matryoshka SwitchTab
    try:
        nesting_list = [32, 64, 96, 128]  # Matching d_model=128
        matryoshka_model = PaperExactSwitchTabMatryoshka(
            embedding_config=config.embedding_config,
            backbone_config=config.backbone_config,
            output_dim=config.output_dim,
            nesting_list=nesting_list,
            temperature=-1.0  # Disable temperature
        )
        matryoshka_success = test_model_overfitting(matryoshka_model, X, y, "PaperExactSwitchTabMatryoshka")
    except Exception as e:
        print(f"❌ Matryoshka SwitchTab failed with error: {e}")
        matryoshka_success = False
    
    # Summary
    print(f"\n{'='*70}")
    print("SANITY CHECK RESULTS:")
    print(f"Regular SwitchTab:    {'✅ PASS' if regular_success else '❌ FAIL'}")
    print(f"Matryoshka SwitchTab: {'✅ PASS' if matryoshka_success else '❌ FAIL'}")
    print(f"{'='*70}")
    
    if regular_success and matryoshka_success:
        print("🎉 Both models can overfit! The issue is likely with loss scaling or training setup.")
    elif regular_success and not matryoshka_success:
        print("🔍 Matryoshka model has implementation issues - focus on loss function or architecture.")
    elif not regular_success and not matryoshka_success:
        print("🚨 Both models have fundamental issues - check basic implementation.")
    else:
        print("🤔 Unexpected result - regular model fails but Matryoshka works?")

if __name__ == "__main__":
    main()
