import torch
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import pandas as pd
import argparse
import os
from scipy.stats import ks_2samp
import pickle
from pathlib import Path

from simple_switchtab import SwitchTabModel
from benchmark.datasets import load_higgs

# --- Hyperparameters from main.tex ---
BATCH_SIZE = 128 # Greatly increased for better GPU utilization. You may need to tune the learning rate.
PRETRAIN_LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 0.003
PRETRAIN_EPOCHS = 130
FINETUNE_EPOCHS = 135
CORRUPTION_RATIO = 0.3
ALPHA = 1.0  # Balance parameter for pre-training losses
GRAD_CLIP_VALUE = 0.5 # Gradient clipping value
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FINETUNE_SCHEDULER = True  # Whether to use OneCycleLR for fine-tuning
MAX_DATASET_SIZE = 100000

def find_best_num_heads(feature_size: int, preferred_num_heads: int = 2) -> int:
    """
    Find the best number of heads for a given feature size.
    It will try to use the preferred_num_heads if it's a divisor.
    Otherwise, it will find the smallest possible divisor greater than preferred_num_heads.
    """
    if feature_size % preferred_num_heads == 0:
        return preferred_num_heads
    
    for i in range(preferred_num_heads + 1, feature_size + 1):
        if feature_size % i == 0:
            return i
            
    return 1 # Fallback, should not happen if feature_size > 1

class PretrainSwitchTabDataset(Dataset):
    """
    Dataset for pre-training SwitchTab. It returns an original sample,
    a corrupted version, and its label. Corruption is done on-the-fly.
    """
    def __init__(self, features, labels, full_train_features, corruption_ratio):
        self.features = features
        self.labels = labels
        self.full_train_features = full_train_features
        self.corruption_ratio = corruption_ratio
        
        self.n_features = features.shape[1]
        self.n_train_samples = full_train_features.shape[0]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x_original = self.features[idx]
        y_label = self.labels[idx]

        # Vectorized feature corruption for a single sample
        n_features_to_corrupt = int(self.n_features * self.corruption_ratio)
        
        if n_features_to_corrupt == 0:
            return x_original, x_original.clone(), y_label

        corruption_indices = torch.randperm(self.n_features)[:n_features_to_corrupt]
        
        # Sample replacement values from the full training set
        random_rows = torch.randint(0, self.n_train_samples, (n_features_to_corrupt,))
        
        x_corrupted = x_original.clone()
        x_corrupted[corruption_indices] = self.full_train_features[random_rows, corruption_indices]

        return x_original, x_corrupted, y_label

def sample_large_dataset(file_path: str, max_size: int = MAX_DATASET_SIZE, random_state: int = 42) -> pd.DataFrame:
    """Smart sampling - get dataset size first, then sample appropriately with exact row count"""
    
    try:
        total_rows = len(pd.read_csv(file_path, sep='\t', usecols=[0]))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"Error reading {file_path}. It might be empty or not found.")
        return pd.DataFrame()

    print(f"Dataset has {total_rows:,} rows")
    
    if total_rows <= max_size:
        df = pd.read_csv(file_path, sep='\t')
        print(f"Loaded all {len(df):,} rows")
        return df
    
    print(f"Sampling exactly {max_size:,} rows from {total_rows:,} total rows")
    
    np.random.seed(random_state)
    rows_to_keep = sorted(np.random.choice(range(1, total_rows + 1), size=max_size, replace=False))
    
    all_rows = set(range(1, total_rows + 1))
    rows_to_skip = sorted(list(all_rows - set(rows_to_keep)))
    
    df = pd.read_csv(file_path, sep='\t', skiprows=rows_to_skip)
    
    print(f"Sampled exactly {len(df):,} rows from {total_rows:,} total rows")
    return df

def calculate_ks_statistic(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            x_batch_seq = x_batch.unsqueeze(1)
            encoded = model.encoder(x_batch_seq).squeeze(1)
            logits = model.predictor(encoded)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probabilities of the positive class
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    class_0_probs = all_probs[all_labels == 0]
    class_1_probs = all_probs[all_labels == 1]
    
    if len(class_0_probs) > 0 and len(class_1_probs) > 0:
        ks_stat, p_value = ks_2samp(class_0_probs, class_1_probs)
        return ks_stat, p_value
    return 0, 1

def get_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            x_batch_seq = x_batch.unsqueeze(1)
            encoded = model.encoder(x_batch_seq).squeeze(1)
            embeddings.append(encoded.cpu().numpy())
    return np.concatenate(embeddings, axis=0)

def main(args):
    """
    Main function to train and evaluate the SwitchTab model.
    """
    print(f"Using device: {DEVICE}")

    # Optimize for Tensor Cores on RTX GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        print("TensorFloat-32 execution enabled for faster performance on compatible GPUs.")

    # --- 1. Load and Prepare Data ---
    if args.train_csv and args.test_csv and args.target:
        print("Loading custom dataset...")
        
        dataset_name = Path(args.train_csv).stem.split('_train')[0]
        
        train_val_df = sample_large_dataset(args.train_csv)
        test_df = sample_large_dataset(args.test_csv)

        if train_val_df.empty or test_df.empty:
            print("Could not load data. Exiting.")
            return

        print("Preprocessing data...")
        
        y_train_val = train_val_df[args.target]
        X_train_val = train_val_df.drop(columns=[args.target])
        y_test = test_df[args.target]
        X_test = test_df.drop(columns=[args.target])

        train_cols = X_train_val.columns
        test_cols = X_test.columns
        shared_cols = list(set(train_cols) & set(test_cols))
        X_train_val = X_train_val[shared_cols]
        X_test = X_test[shared_cols]

        missing_ratios = X_train_val.isnull().sum() / len(X_train_val)
        cols_to_drop = missing_ratios[missing_ratios > 0.95].index
        X_train_val = X_train_val.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} columns with >95% missing values: {list(cols_to_drop)}")

        categorical_features = X_train_val.select_dtypes(include=['object', 'category']).columns
        numerical_features = X_train_val.select_dtypes(include=np.number).columns
        print(f"Found {len(categorical_features)} categorical features and {len(numerical_features)} numerical features.")

        median_values = X_train_val[numerical_features].median()
        X_train_val.fillna(median_values, inplace=True)
        X_test.fillna(median_values, inplace=True)
            
        for col in categorical_features:
            X_train_val[col] = X_train_val[col].astype('category').cat.codes
            X_test[col] = X_test[col].astype('category').cat.codes
            X_train_val[col] += 1
            X_test[col] += 1

        le = LabelEncoder()
        y_train_val_encoded = le.fit_transform(y_train_val)
        y_test_encoded = le.transform(y_test)

        X_train_df, X_val_df, y_train_np, y_val_np = train_test_split(
            X_train_val, y_train_val_encoded, test_size=0.2, random_state=42, stratify=y_train_val_encoded
        )
        y_test_np = y_test_encoded

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_df.values)
        X_val_scaled = scaler.transform(X_val_df.values)
        X_test_scaled = scaler.transform(X_test.values)

        X_train = torch.from_numpy(X_train_scaled).float()
        y_train = torch.from_numpy(y_train_np).long()
        X_val = torch.from_numpy(X_val_scaled).float()
        y_val = torch.from_numpy(y_val_np).long()
        X_test = torch.from_numpy(X_test_scaled).float()
        y_test = torch.from_numpy(y_test_np).long()
    else:
        dataset_name = "higgs"
        print("Loading Higgs dataset...")
        data, label, _, _, _, _, _ = load_higgs()
        
        # Replace -999.0 placeholders with 0
        data = data.replace(-999.0, 0).fillna(0)
        
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            data, label.values, test_size=0.2, random_state=42, stratify=label.values
        )

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train_np.values)
        X_val_scaled = scaler.transform(X_val_np.values)
        X_test_scaled = scaler.transform(X_test_np.values)

        X_train = torch.from_numpy(X_train_scaled).float()
        y_train = torch.from_numpy(y_train_np).long()
        X_val = torch.from_numpy(X_val_scaled).float()
        y_val = torch.from_numpy(y_val_np).long()
        X_test = torch.from_numpy(X_test_scaled).float()
        y_test = torch.from_numpy(y_test_np).long()

    # --- 2. Create Datasets and DataLoaders ---
    num_workers = min(16, os.cpu_count()) if os.cpu_count() else 4
    print(f"Using {num_workers} workers for data loading.")

    # Pre-training dataset with on-the-fly corruption
    pretrain_dataset = PretrainSwitchTabDataset(X_train, y_train, X_train, CORRUPTION_RATIO)
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True
    )

    # Fine-tuning and test datasets (no corruption needed)
    finetune_dataset = TensorDataset(X_train, y_train)
    finetune_loader = DataLoader(
        finetune_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    
    feature_size = X_train.shape[1]
    num_classes = len(torch.unique(y_train))
    
    print(f"Data loaded: {len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"Feature size: {feature_size}, Num classes: {num_classes}")

    # --- 3. Initialize Model and Optimizer ---
    NUM_HEADS = find_best_num_heads(feature_size)
    print(f"Using {NUM_HEADS} attention heads.")
    model = SwitchTabModel(feature_size=feature_size, num_classes=num_classes, num_heads=NUM_HEADS).to(DEVICE)

    # Compile the model for a significant performance boost with PyTorch 2.0+
    if hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)

    pretrain_optimizer = optim.RMSprop(model.parameters(), lr=PRETRAIN_LEARNING_RATE)
    pretrain_scheduler = optim.lr_scheduler.OneCycleLR(
        pretrain_optimizer, max_lr=PRETRAIN_LEARNING_RATE, steps_per_epoch=len(pretrain_loader), epochs=PRETRAIN_EPOCHS
    )
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    # --- 4. Pre-training (Semi-supervised) ---
    print("\n--- Starting Pre-training ---")
    model.train()
    for epoch in range(PRETRAIN_EPOCHS):
        total_loss, total_recon_loss, total_cls_loss = 0, 0, 0
        
        for i, ((x1_orig, x1_corr, y1_batch), (x2_orig, x2_corr, y2_batch)) in enumerate(zip(pretrain_loader, pretrain_loader)):
            x1_orig, x1_corr, y1_batch = x1_orig.to(DEVICE), x1_corr.to(DEVICE), y1_batch.to(DEVICE)
            x2_orig, x2_corr, y2_batch = x2_orig.to(DEVICE), x2_corr.to(DEVICE), y2_batch.to(DEVICE)

            # --- Forward Pass ---
            x1_corrupted_seq = x1_corr.unsqueeze(1)
            x2_corrupted_seq = x2_corr.unsqueeze(1)

            outputs = model(x1_corrupted_seq, x2_corrupted_seq)
            x1_rec, x2_rec, x1_sw, x2_sw, z1_logits, z2_logits = [o.squeeze(1) if o.ndim == 3 else o for o in outputs]

            # --- Loss Calculation ---
            recon_loss = (mse_loss_fn(x1_rec, x1_orig) + 
                          mse_loss_fn(x2_rec, x2_orig) +
                          mse_loss_fn(x1_sw, x1_orig) + 
                          mse_loss_fn(x2_sw, x2_orig))
            
            cls_loss = ce_loss_fn(z1_logits, y1_batch) + ce_loss_fn(z2_logits, y2_batch)

            loss = recon_loss + ALPHA * cls_loss
            
            # --- Backward Pass and Optimization ---
            pretrain_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            pretrain_optimizer.step()
            pretrain_scheduler.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()

        num_batches = len(pretrain_loader)
        if num_batches > 0:
            print(f"Epoch [{epoch+1}/{PRETRAIN_EPOCHS}], Total Loss: {total_loss/num_batches:.4f}, "
                  f"Recon Loss: {total_recon_loss/num_batches:.4f}, CLS Loss: {total_cls_loss/num_batches:.4f}")

    # --- 5. Fine-tuning (Supervised) ---
    print("\n--- Starting Fine-tuning ---")
    finetune_optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.predictor.parameters()), 
        lr=FINETUNE_LEARNING_RATE
    )
    if USE_FINETUNE_SCHEDULER:
        finetune_scheduler = optim.lr_scheduler.OneCycleLR(
            finetune_optimizer, max_lr=FINETUNE_LEARNING_RATE, steps_per_epoch=len(finetune_loader), epochs=FINETUNE_EPOCHS
        )
    
    best_val_ks = -1
    model_save_path = f"best_model_{dataset_name}.pt"

    model.train()
    for epoch in range(FINETUNE_EPOCHS):
        total_loss = 0
        for x_batch, y_batch in finetune_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            x_batch_seq = x_batch.unsqueeze(1)
            encoded = model.encoder(x_batch_seq).squeeze(1)
            logits = model.predictor(encoded)
            
            loss = ce_loss_fn(logits, y_batch)
            
            finetune_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            finetune_optimizer.step()
            if USE_FINETUNE_SCHEDULER:
                finetune_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(finetune_loader)
        print(f"Epoch [{epoch+1}/{FINETUNE_EPOCHS}], Fine-tuning Loss: {avg_loss:.6f}")

        # Validation step
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                
                x_batch_seq = x_batch.unsqueeze(1)
                encoded = model.encoder(x_batch_seq).squeeze(1)
                logits = model.predictor(encoded)
                
                loss = ce_loss_fn(logits, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        ks_stat, _ = calculate_ks_statistic(model, val_loader, DEVICE)
        print(f"Epoch [{epoch+1}/{FINETUNE_EPOCHS}], Val Loss: {avg_val_loss:.6f}, Val Accuracy: {val_accuracy:.2f}%, Val KS: {ks_stat:.4f}")
        
        if ks_stat > best_val_ks:
            best_val_ks = ks_stat
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with KS: {best_val_ks:.4f}")
        
        model.train()
        
    # --- 6. Load Best Model and Save Embeddings ---
    print("\n--- Loading best model and saving embeddings ---")
    model.load_state_dict(torch.load(model_save_path))

    train_embeddings = get_embeddings(model, finetune_loader, DEVICE)
    val_embeddings = get_embeddings(model, val_loader, DEVICE)
    test_embeddings = get_embeddings(model, test_loader, DEVICE)

    with open(f"embeddings_{dataset_name}_train.pkl", "wb") as f:
        pickle.dump(train_embeddings, f)
    with open(f"embeddings_{dataset_name}_val.pkl", "wb") as f:
        pickle.dump(val_embeddings, f)
    with open(f"embeddings_{dataset_name}_test.pkl", "wb") as f:
        pickle.dump(test_embeddings, f)
        
    print("Embeddings saved successfully.")

    # --- 7. Evaluation ---
    print("\n--- Evaluating Model ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            x_batch_seq = x_batch.unsqueeze(1)
            encoded = model.encoder(x_batch_seq).squeeze(1)
            logits = model.predictor(encoded)
            
            _, predicted = torch.max(logits.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    ks_stat, _ = calculate_ks_statistic(model, test_loader, DEVICE)
    print(f"\nTest Accuracy: {accuracy:.2f}%, Test KS: {ks_stat:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SwitchTab model on a custom dataset.")
    parser.add_argument("--train_csv", type=str, default=None, help="Path to the training/validation CSV file (tab-separated).")
    parser.add_argument("--test_csv", type=str, default=None, help="Path to the test CSV file (tab-separated).")
    parser.add_argument("--target", type=str, default=None, help="Name of the target column.")
    
    args = parser.parse_args()
    main(args)
