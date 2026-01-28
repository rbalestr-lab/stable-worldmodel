import argparse
import fcntl
import gc
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Assuming these are available in your environment
from stable_worldmodel.finance_data.download import load_market_data

# ==========================================
# PART 1: INFRASTRUCTURE
# ==========================================

# Set SLURM_NTASKS_PER_NODE if needed
if "SLURM_NTASKS" in os.environ and "SLURM_NTASKS_PER_NODE" not in os.environ:
    if "SLURM_NNODES" in os.environ:
        ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        nnodes = int(os.environ.get("SLURM_NNODES", "1"))
        os.environ["SLURM_NTASKS_PER_NODE"] = str(ntasks // nnodes)
    else:
        os.environ["SLURM_NTASKS_PER_NODE"] = os.environ.get("SLURM_NTASKS", "1")

def load_results(results_file="results.json", max_retries=10, retry_delay=0.1):
    """Load results from JSON file with file locking."""
    results_path = Path(results_file)
    if not results_path.exists():
        return {}

    for attempt in range(max_retries):
        try:
            with open(results_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    return json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError, json.JSONDecodeError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return {}
    return {}

def save_results(results, results_file="results.json", max_retries=10, retry_delay=0.1):
    """Save results to JSON file with file locking."""
    results_path = Path(results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = results_path.with_suffix('.tmp')

    for attempt in range(max_retries):
        try:
            with open(temp_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(results, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            temp_path.replace(results_path)
            return
        except (IOError, OSError):
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise

def get_all_tickers():
    """Load all available tickers from NYSE and NASDAQ."""
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    finance_data_dir = os.path.join(root_dir, "stable_worldmodel", "finance_data")
    
    try:
        df1 = pd.read_csv(os.path.join(finance_data_dir, "nyse.csv"))
        df2 = pd.read_csv(os.path.join(finance_data_dir, "nasdaq.csv"))
        df1["Symbol"] = df1["Symbol"].str.strip()
        df2["Symbol"] = df2["Symbol"].str.strip()
        all_tickers = sorted(set(df1["Symbol"].tolist() + df2["Symbol"].tolist()))
        return all_tickers
    except Exception as e:
        print(f"Warning: Could not load ticker CSVs ({e}). Defaulting to AAPL/MSFT.")
        return ["AAPL", "MSFT"]

# ==========================================
# PART 2: FINANCIAL LOGIC (RAM-BASED FAST VERSION)
# ==========================================

def compute_ema(data: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema

def rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros_like(data)
    for t in range(window, len(data)):
        result[t] = np.std(data[t - window : t], axis=0)
    return result

def add_technical_features(data: np.ndarray) -> np.ndarray:
    """Add technical indicators (EMAs, momentum, volatility)."""
    # Pad if window is too small to calculate features
    if data.shape[0] < 30:
        return np.pad(data, ((0, 30-data.shape[0]), (0,0), (0, 9)), 'edge')

    close = data[:, :, 3]
    high = data[:, :, 1]
    low = data[:, :, 2]
    volume = data[:, :, 4]

    ema_5 = compute_ema(close, span=5)
    ema_15 = compute_ema(close, span=15)
    ema_30 = compute_ema(close, span=30)
    
    returns_1 = np.diff(close, axis=0, prepend=close[:1])
    # Fix: Use padding instead of roll to avoid wraparound
    returns_5 = close - np.concatenate([np.repeat(close[:1], 5, axis=0), close[:-5]], axis=0)
    returns_10 = close - np.concatenate([np.repeat(close[:1], 10, axis=0), close[:-10]], axis=0)
    
    volatility = rolling_std(returns_1, window=15)
    volume_ma = compute_ema(volume, span=10)
    volume_ratio = volume / (volume_ma + 1e-8)
    hl_range = (high - low) / (close + 1e-8)

    new_features = np.stack([
        ema_5, ema_15, ema_30, returns_1, returns_5, returns_10,
        volatility, volume_ratio, hl_range
    ], axis=2)

    enhanced_data = np.concatenate([data, new_features], axis=2)
    # Fix: Handle NaN, Inf, and clip extreme values
    enhanced_data = np.nan_to_num(enhanced_data, nan=0.0, posinf=1e6, neginf=-1e6)
    enhanced_data = np.clip(enhanced_data, -1e6, 1e6)
    
    # Debug: Check for remaining issues
    if np.isnan(enhanced_data).any():
        print(f"WARNING: NaN found in enhanced_data after cleaning")
    if np.isinf(enhanced_data).any():
        print(f"WARNING: Inf found in enhanced_data after cleaning")
    
    return enhanced_data

class FinancialDataset(Dataset):
    """
    Standard Dataset (RAM for Raw Data, CPU for Features).
    This mimics the original script's strategy:
    - Load RAW floats (Open, High, Low, Close, Vol) into RAM. (Small footprint)
    - Compute complex features (EMAs, RSI) on-the-fly. (Saves 50GB+ RAM)
    """
    def __init__(self, data, window_size=30, prediction_horizon=30, time_sampling_rate=1):
        # We store the RAW data in RAM. 
        # For 3600 stocks over 2 years, this is about 12GB of RAM.
        print(f"\nDataset initialization:")
        print(f"  Raw data shape: {data.shape}")
        print(f"  Has NaN: {np.isnan(data).any()}")
        print(f"  Has Inf: {np.isinf(data).any()}")
        print(f"  Data range: [{np.min(data):.2f}, {np.max(data):.2f}]")
        
        self.data = data.astype(np.float32)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.num_stocks = data.shape[1]
        self.time_sampling_rate = time_sampling_rate

        # Pre-compute returns once (lightweight)
        close_prices = data[:, :, 3]
        print(f"  Close prices range: [{np.min(close_prices):.2f}, {np.max(close_prices):.2f}]")
        print(f"  Close prices has zeros: {(close_prices == 0).sum()} out of {close_prices.size}")
        
        future_close = np.roll(close_prices, -prediction_horizon, axis=0)
        self.returns = ((future_close - close_prices) / (close_prices + 1e-8))
        
        # Clip returns to reasonable range
        self.returns = np.clip(self.returns, -1.0, 1.0)
        print(f"  Returns range: [{np.min(self.returns):.4f}, {np.max(self.returns):.4f}]")
        print(f"  Returns has NaN: {np.isnan(self.returns).any()}")

        # Valid indices
        all_valid = list(range(window_size, len(data) - prediction_horizon))
        self.valid_indices = all_valid[::time_sampling_rate]
        print(f"  Valid samples: {len(self.valid_indices)}")

        # Determine feature count by running one dummy sample
        dummy_window = data[:window_size, :1, :]
        enhanced_dummy = add_technical_features(dummy_window)
        self.enhanced_num_features = enhanced_dummy.shape[2]
        print(f"  Enhanced features: {self.enhanced_num_features}\n")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. Random Stock Sample (No Cartesian Product)
        stock_idx = np.random.randint(self.num_stocks)
        time_idx = self.valid_indices[idx]
        
        start_t = time_idx - self.window_size
        
        # 2. Slice from RAM (Instant, nanoseconds)
        # This is the key speedup over the "Lazy" approach
        historical_raw = self.data[start_t:time_idx, stock_idx, :]
        
        # 3. Compute Features on-the-fly (Saves ~50GB RAM vs pre-computing)
        historical_raw_3d = historical_raw[:, None, :]
        historical_enhanced = add_technical_features(historical_raw_3d)
        features = historical_enhanced[:, 0, :].flatten()
        
        # Normalize features (z-score with robust stats)
        feature_mean = features.mean()
        feature_std = features.std() + 1e-8
        features = (features - feature_mean) / feature_std
        features = np.clip(features, -10, 10)  # Clip to reasonable range
        
        target = self.returns[time_idx, stock_idx]
        
        # Fix: Clip target returns to prevent extreme values
        target = np.clip(target, -1.0, 1.0)
        
        return torch.from_numpy(features.astype(np.float32)), torch.tensor(target, dtype=torch.float32), stock_idx

# ==========================================
# PART 3: MODELING
# ==========================================

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, window_size):
        batch_size = x.shape[0]
        num_features = x.shape[1] // window_size
        x = x.reshape(batch_size, window_size, num_features)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_proj(x).squeeze(-1)

class FinancialLightningModule(pl.LightningModule):
    def __init__(self, input_dim, window_size, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleTransformer(input_dim=input_dim)
        self.window_size = window_size
        self.criterion = nn.MSELoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x, self.window_size)

    def training_step(self, batch, batch_idx):
        features, targets, _ = batch
        
        # Debug first batch
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"\n[Training Debug - First Batch]")
            print(f"  Features shape: {features.shape}")
            print(f"  Features has NaN: {torch.isnan(features).any().item()}")
            print(f"  Features has Inf: {torch.isinf(features).any().item()}")
            print(f"  Features range: [{features.min().item():.4f}, {features.max().item():.4f}]")
            print(f"  Targets has NaN: {torch.isnan(targets).any().item()}")
            print(f"  Targets range: [{targets.min().item():.4f}, {targets.max().item():.4f}]\n")
        
        preds = self(features)
        
        if batch_idx == 0 and self.current_epoch == 0:
            print(f"[Predictions Debug - First Batch]")
            print(f"  Predictions has NaN: {torch.isnan(preds).any().item()}")
            print(f"  Predictions has Inf: {torch.isinf(preds).any().item()}")
            print(f"  Predictions range: [{preds.min().item():.4f}, {preds.max().item():.4f}]\n")
        
        loss = self.criterion(preds, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, targets, _ = batch
        preds = self(features)
        loss = self.criterion(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append({"preds": preds.detach().cpu(), "targets": targets.detach().cpu()})
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])
        ic, _ = spearmanr(all_preds.numpy(), all_targets.numpy())
        self.log("val_ic", ic, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        features, targets, _ = batch
        preds = self(features)
        loss = self.criterion(preds, targets)
        self.log("test_loss", loss)
        self.test_step_outputs.append({"preds": preds.detach().cpu(), "targets": targets.detach().cpu()})
        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
        ic, _ = spearmanr(all_preds.numpy(), all_targets.numpy())
        self.log("test_ic", ic)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

# ==========================================
# PART 4: MAIN EXECUTION
# ==========================================

def get_hyperparams_dict(args):
    return {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "train_start": args.train_start,
        "train_end": args.train_end
    }

def main(args):
    results_file = args.results_file
    results = load_results(results_file)
    hyperparams = get_hyperparams_dict(args)
    hyperparams_key = tuple(sorted(hyperparams.items()))
    
    if not args.force_rerun and str(hyperparams_key) in str(results):
         print("Configuration found in results. Skipping...")
         return

    # 1. Get Tickers
    all_tickers = get_all_tickers()
    print(f"Loading metadata for {len(all_tickers)} tickers...")

    # 2. LOAD RAW DATA INTO RAM (FASTEST METHOD)
    # The load takes ~30s once, but then accessing batches is instant.
    print(f"Loading Train Data ({args.train_start} to {args.train_end})...")
    train_data_raw = load_market_data(
        "train.h5",
        tickers=all_tickers,
        start_time=args.train_start,
        end_time=args.train_end,
        freq="1min"
    )
    
    print(f"Loading Test Data ({args.test_start} to {args.test_end})...")
    test_data_raw = load_market_data(
        "test.h5",
        tickers=all_tickers,
        start_time=args.test_start,
        end_time=args.test_end,
        freq="1min"
    )

    # 3. Initialize Datasets (Passing the RAM data)
    print("Initializing Datasets...")
    
    # Drop timestamps with NaN values
    print("\nCleaning Train Data...")
    train_nan_mask = np.isnan(train_data_raw).any(axis=(1, 2))  # Check each timestamp
    train_nan_count = train_nan_mask.sum()
    print(f"  Dropping {train_nan_count} timestamps with NaN out of {len(train_data_raw)}")
    train_data_raw = train_data_raw[~train_nan_mask]
    print(f"  Remaining timestamps: {len(train_data_raw)}")
    
    print("\nCleaning Test Data...")
    test_nan_mask = np.isnan(test_data_raw).any(axis=(1, 2))
    test_nan_count = test_nan_mask.sum()
    print(f"  Dropping {test_nan_count} timestamps with NaN out of {len(test_data_raw)}")
    test_data_raw = test_data_raw[~test_nan_mask]
    print(f"  Remaining timestamps: {len(test_data_raw)}\n")
    
    train_dataset = FinancialDataset(
        train_data_raw, 
        window_size=args.window_size,
        time_sampling_rate=args.time_sampling_rate
    )
    test_dataset = FinancialDataset(
        test_data_raw, 
        window_size=args.window_size,
        time_sampling_rate=args.time_sampling_rate
    )
    
    # Clean up outer references to free memory for the Training process
    del train_data_raw, test_data_raw
    gc.collect()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 4. Setup Model
    model = FinancialLightningModule(
        input_dim=train_dataset.enhanced_num_features,
        window_size=args.window_size,
        lr=args.lr
    )

    # 5. Setup Trainer
    checkpoint_dir = f"./checkpoints/finance-{args.train_start}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_ic", 
        mode="max", 
        save_top_k=1, 
        filename="best-finance-model"
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=False,  # Disable logger, use print statements
        gradient_clip_val=1.0,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        enable_progress_bar=True
    )

    print("\n" + "="*50)
    print("TRAINING STARTED")
    print("="*50)
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*50)
    print("TESTING ON VALIDATION SET")
    print("="*50)
    
    trainer.test(model, val_loader, ckpt_path="best")
    
    final_metrics = trainer.callback_metrics
    test_ic = final_metrics.get("test_ic", 0.0)
    test_loss = final_metrics.get("test_loss", 0.0)
    
    if isinstance(test_ic, torch.Tensor):
        test_ic = test_ic.item()
    if isinstance(test_loss, torch.Tensor):
        test_loss = test_loss.item()
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Test IC (Spearman): {test_ic:.4f}")
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print("="*50 + "\n")
    
    result_entry = {
        "hyperparams": hyperparams, 
        "test_ic": test_ic,
        "test_loss": test_loss,
        "timestamp": datetime.now().isoformat()
    }
    
    results = load_results(results_file)
    if "financial_transformer" not in results:
        results["financial_transformer"] = []
    results["financial_transformer"].append(result_entry)
    save_results(results, results_file)
    print(f"✓ Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--train_start", type=str, default="2022-01-03")
    parser.add_argument("--train_end", type=str, default="2023-12-31")
    # parser.add_argument("--train_end", type=str, default="2022-01-04")
    parser.add_argument("--test_start", type=str, default="2024-01-02")
    parser.add_argument("--test_end", type=str, default="2024-03-31")
    # parser.add_argument("--test_end", type=str, default="2024-01-03")
    
    # Model args
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)  # Reduced from 1e-4 for stability
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--time_sampling_rate", type=int, default=1)
    
    # Infra args
    parser.add_argument("--results_file", type=str, default="financial_results.json")
    parser.add_argument("--force_rerun", action="store_true")
    
    args = parser.parse_args()
    main(args)