"""Supervised baseline models for financial return prediction.

Train linear and transformer models to predict 30-minute returns using minute-level data.
Computes per-stock Information Coefficient (IC) as evaluation metric.

Uses stable_worldmodel's World interface and data infrastructure.

Usage:
    # Train on all available tickers (default: ~3600 stocks from NYSE/NASDAQ)
    # No --tickers argument = uses all stocks automatically
    python scripts/train/supervised_baseline.py --model both \\
        --train-start 2022-01-03 --train-end 2023-12-31 \\
        --test-start 2024-01-02 --test-end 2024-03-31

    # Train linear model on specific tickers only
    python scripts/train/supervised_baseline.py --model linear --tickers AAPL MSFT GOOGL \\
        --train-start 2023-01-01 --train-end 2023-12-31 \\
        --test-start 2024-01-02 --test-end 2024-06-30

    # Train transformer with custom hyperparameters and memory settings
    python scripts/train/supervised_baseline.py --model transformer --tickers AAPL MSFT \\
        --train-start 2022-01-03 --train-end 2023-12-31 \\
        --test-start 2024-01-02 --test-end 2024-03-31 \\
        --epochs 50 --batch-size 64 --gradient-accumulation-steps 4 --lr 1e-4

    # Reduce memory further with aggressive time sampling
    python scripts/train/supervised_baseline.py --model both \\
        --train-start 2022-01-03 --train-end 2023-12-31 \\
        --test-start 2024-01-02 --test-end 2024-03-31 \\
        --time-sampling-rate 10 --batch-size 32 --gradient-accumulation-steps 8
"""

import argparse
import gc
import os
from datetime import datetime

import matplotlib


matplotlib.use("Agg")  # Use non-interactive backend for compatibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from scipy.stats import spearmanr
from sklearn.linear_model import SGDRegressor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stable_worldmodel.finance_data.download import load_market_data


def get_all_tickers():
    """Load all available tickers from NYSE and NASDAQ."""
    finance_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "stable_worldmodel",
        "finance_data",
    )
    df1 = pd.read_csv(os.path.join(finance_data_dir, "nyse.csv"))
    df2 = pd.read_csv(os.path.join(finance_data_dir, "nasdaq.csv"))
    df1["Symbol"] = df1["Symbol"].str.strip()
    df2["Symbol"] = df2["Symbol"].str.strip()
    all_tickers = sorted(set(df1["Symbol"].tolist() + df2["Symbol"].tolist()))
    return all_tickers


def add_technical_features(data: np.ndarray) -> np.ndarray:
    """Add technical indicators (EMAs, momentum, volatility) to raw OHLCV data.

    Args:
        data: (T, num_stocks, features) where features = [open, high, low, close, volume, trade_count, vwap]

    Returns:
        Enhanced data with additional technical features
    """
    close = data[:, :, 3]
    high = data[:, :, 1]
    low = data[:, :, 2]
    volume = data[:, :, 4]

    # Compute EMAs with different windows
    ema_5 = compute_ema(close, span=5)
    ema_15 = compute_ema(close, span=15)
    ema_30 = compute_ema(close, span=30)

    # Compute returns at different horizons
    returns_1 = np.diff(close, axis=0, prepend=close[:1])
    returns_5 = close - np.roll(close, 5, axis=0)
    returns_10 = close - np.roll(close, 10, axis=0)

    # Compute rolling volatility (std of returns)
    volatility = rolling_std(returns_1, window=15)

    # Volume momentum
    volume_ma = compute_ema(volume, span=10)
    volume_ratio = volume / (volume_ma + 1e-8)

    # High-Low range
    hl_range = (high - low) / (close + 1e-8)

    # Stack all features
    new_features = np.stack(
        [
            ema_5,
            ema_15,
            ema_30,
            returns_1,
            returns_5,
            returns_10,
            volatility,
            volume_ratio,
            hl_range,
        ],
        axis=2,
    )

    # Concatenate with original features
    enhanced_data = np.concatenate([data, new_features], axis=2)

    # Handle NaN values (from rolling computations)
    enhanced_data = np.nan_to_num(enhanced_data, nan=0.0)

    return enhanced_data


def compute_ema(data: np.ndarray, span: int) -> np.ndarray:
    """Compute exponential moving average along time axis."""
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]

    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]

    return ema


def rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation."""
    result = np.zeros_like(data)
    for t in range(window, len(data)):
        result[t] = np.std(data[t - window : t], axis=0)
    return result


class FinancialDataset(Dataset):
    """Memory-efficient PyTorch Dataset for financial time series prediction.

    KEY FIX: Does NOT create Cartesian product (time × stock).
    Instead, samples one random stock per time step, reducing dataset size by ~3600×.
    Technical features are computed on-the-fly to avoid storing enhanced data.
    """

    def __init__(
        self,
        data: np.ndarray,  # Shape: (T, num_stocks, features)
        symbols: list[str],
        window_size: int = 30,
        prediction_horizon: int = 30,
        time_sampling_rate: int = 1,  # Sample every Nth minute
    ):
        """Initialize financial dataset.

        Args:
            data: Market data array (T, num_stocks, features) - RAW data only (no tech features)
            symbols: List of stock symbols
            window_size: Number of historical minutes to use
            prediction_horizon: Minutes ahead to predict returns
            time_sampling_rate: Sample every Nth time step (1=all, 5=every 5 min)
        """
        # Store RAW data only (no enhanced features)
        self.data = data.astype(np.float32)  # Force float32 immediately
        self.symbols = symbols
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.num_stocks = data.shape[1]
        self.num_features = data.shape[2]  # Raw features only (~7)
        self.time_sampling_rate = time_sampling_rate

        # Pre-compute returns for efficiency (lightweight)
        close_prices = data[:, :, 3].astype(np.float32)  # Close is 4th feature
        future_close = np.roll(close_prices, -prediction_horizon, axis=0)
        self.returns = ((future_close - close_prices) / (close_prices + 1e-8)).astype(np.float32)

        # Valid time indices (have enough history and future)
        all_valid = list(range(window_size, len(data) - prediction_horizon))
        self.valid_indices = all_valid[::time_sampling_rate]  # Sample every Nth

        # Determine enhanced feature count by computing once
        # Raw features: 7 (open, high, low, close, volume, trade_count, vwap)
        # Tech features added: 9 (ema_5, ema_15, ema_30, returns_1, returns_5, returns_10, volatility, volume_ratio, hl_range)
        # Total enhanced: 16
        dummy_window = data[:window_size, :1, :]  # (window, 1 stock, raw_features)
        enhanced_dummy = add_technical_features(dummy_window)
        self.enhanced_num_features = enhanced_dummy.shape[2]

        logger.info(
            f"Dataset initialized: {len(self.valid_indices)} time samples "
            f"(1 random stock per sample), {self.num_stocks} stocks total, "
            f"{self.num_features} raw features → {self.enhanced_num_features} enhanced features (computed on-the-fly)"
        )
        logger.info(f"Memory footprint reduced by ~{self.num_stocks}× vs Cartesian product")

    def __len__(self):
        # KEY FIX: Length is just number of time samples, NOT (time × stock)
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get a single sample.

        Returns:
            features: (window_size * enhanced_features,) flattened features with tech indicators
            target: scalar future return
            stock_idx: which stock this sample is for
            time_idx: which time this sample is for
        """
        # KEY FIX: Sample ONE random stock per time step
        stock_idx = np.random.randint(self.num_stocks)
        time_idx = self.valid_indices[idx]

        # Get historical window for this stock
        start_t = time_idx - self.window_size
        historical_raw = self.data[start_t:time_idx, stock_idx, :]  # (window, raw_features)

        # Compute technical features ON-THE-FLY for this window only
        # Add a dummy stock dimension: (window, features) -> (window, 1, features)
        historical_raw_3d = historical_raw[:, None, :]
        historical_enhanced = add_technical_features(historical_raw_3d)  # (window, 1, enhanced_features)
        historical_enhanced = historical_enhanced[:, 0, :]  # (window, enhanced_features)

        # Flatten to 1D
        features = historical_enhanced.flatten().astype(np.float32)

        # Get target return
        target = self.returns[time_idx, stock_idx]

        return (
            torch.from_numpy(features),  # Already float32, avoid copy
            torch.tensor(target, dtype=torch.float32),
            stock_idx,
            time_idx,
        )


class SimpleTransformer(nn.Module):
    """Simple transformer model for time series prediction."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, window_size):
        """Forward pass.

        Args:
            x: (batch, window_size * features)
            window_size: int, number of time steps

        Returns:
            predictions: (batch, 1)
        """
        batch_size = x.shape[0]
        num_features = x.shape[1] // window_size

        # Reshape to (batch, window, features)
        x = x.reshape(batch_size, window_size, num_features)

        # Project to d_model
        x = self.input_proj(x)  # (batch, window, d_model)

        # Pass through transformer
        x = self.transformer(x)  # (batch, window, d_model)

        # Use last time step for prediction
        x = x[:, -1, :]  # (batch, d_model)

        # Project to output
        out = self.output_proj(x)  # (batch, 1)

        return out.squeeze(-1)


def compute_ic_per_stock(predictions, targets, stock_indices, num_stocks):
    """Compute Information Coefficient (Spearman correlation) per stock.

    Args:
        predictions: (N,) array of predictions
        targets: (N,) array of actual returns
        stock_indices: (N,) array indicating which stock each sample belongs to
        num_stocks: total number of stocks

    Returns:
        ic_per_stock: (num_stocks,) array of IC values
        avg_ic: average IC across all stocks
    """
    ic_per_stock = []

    for stock_idx in range(num_stocks):
        mask = stock_indices == stock_idx
        if mask.sum() < 2:  # Need at least 2 samples for correlation
            ic_per_stock.append(0.0)
            continue

        stock_preds = predictions[mask]
        stock_targets = targets[mask]

        # Compute Spearman correlation
        ic, _ = spearmanr(stock_preds, stock_targets)
        if np.isnan(ic) or ic is None:
            ic = 0.0

        ic_per_stock.append(ic)

    ic_per_stock = np.array(ic_per_stock)
    avg_ic = np.mean(np.abs(ic_per_stock))

    return ic_per_stock, avg_ic


def train_linear_model(train_dataset, test_dataset, symbols, batch_size=512):
    """Train a linear (SGD) regression model with mini-batch training.

    Memory optimization: Removed StandardScaler (major memory hog).
    Uses simple mean/std normalization computed incrementally.
    """
    logger.info("Training Linear Model (SGDRegressor)")

    # Create data loader for mini-batch training (reduced batch size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Compute simple normalization stats incrementally (much lighter than StandardScaler)
    logger.info("Computing normalization statistics...")
    running_mean = None
    running_var = None
    n_samples = 0

    for features, _, _, _ in tqdm(train_loader, desc="Computing stats"):
        batch_np = features.numpy()
        batch_mean = batch_np.mean(axis=0)
        batch_var = batch_np.var(axis=0)
        batch_n = len(batch_np)

        if running_mean is None:
            running_mean = batch_mean
            running_var = batch_var
            n_samples = batch_n
        else:
            # Welford's online algorithm for stable mean/variance
            delta = batch_mean - running_mean
            running_mean += delta * batch_n / (n_samples + batch_n)
            running_var = (running_var * n_samples + batch_var * batch_n) / (n_samples + batch_n)
            n_samples += batch_n

    running_std = np.sqrt(running_var + 1e-8)
    logger.info("Normalization stats computed")

    # Initialize SGDRegressor with fit_intercept to handle unnormalized data better
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        learning_rate="adaptive",
        eta0=0.01,
        max_iter=1,
        tol=None,
        random_state=42,
        fit_intercept=True,
    )

    # Train in mini-batches and track losses
    n_epochs = 5
    epoch_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for features, targets, _, _ in tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{n_epochs}"):
            # Simple normalization (no copies like StandardScaler)
            X_batch = (features.numpy() - running_mean) / running_std
            y_batch = targets.numpy()
            model.partial_fit(X_batch, y_batch)

            # Compute batch loss
            preds = model.predict(X_batch)
            loss = np.mean((preds - y_batch) ** 2)
            epoch_loss += loss
            n_batches += 1

            # Clear batch memory
            del X_batch

        avg_epoch_loss = epoch_loss / n_batches
        epoch_losses.append(avg_epoch_loss)
        logger.info(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_epoch_loss:.6f}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_epochs + 1), epoch_losses, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Linear Model Training Loss", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"linear_training_loss_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Training loss plot saved to {plot_path}")
    plt.close()

    # Evaluate on training set STREAMING (avoid accumulating all predictions)
    logger.info("Evaluating on training set (streaming)...")

    # Store per-stock predictions/targets in dict to avoid massive concatenation
    stock_data = {i: {"preds": [], "targets": []} for i in range(len(symbols))}

    for features, targets, stock_idx, _ in tqdm(train_loader, desc="Train evaluation"):
        X_batch = (features.numpy() - running_mean) / running_std
        preds = model.predict(X_batch)

        # Append to per-stock lists
        for i, (pred, target, sid) in enumerate(zip(preds, targets.numpy(), stock_idx.numpy())):
            stock_data[sid]["preds"].append(pred)
            stock_data[sid]["targets"].append(target)

        del X_batch

    # Compute IC per stock from accumulated data
    train_ic_per_stock = []
    for stock_idx in range(len(symbols)):
        preds = stock_data[stock_idx]["preds"]
        targets = stock_data[stock_idx]["targets"]

        if len(preds) < 2:
            train_ic_per_stock.append(0.0)
            continue

        ic, _ = spearmanr(preds, targets)
        if np.isnan(ic) or ic is None:
            ic = 0.0
        train_ic_per_stock.append(ic)

    train_ic_per_stock = np.array(train_ic_per_stock)
    train_avg_ic = np.mean(np.abs(train_ic_per_stock))

    # Clear memory
    del stock_data
    gc.collect()

    logger.info(f"Train IC: {train_avg_ic:.4f}")

    # Evaluate on test set STREAMING
    logger.info("Evaluating on test set (streaming)...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    stock_data = {i: {"preds": [], "targets": []} for i in range(len(symbols))}

    for features, targets, stock_idx, _ in tqdm(test_loader, desc="Test evaluation"):
        X_batch = (features.numpy() - running_mean) / running_std
        preds = model.predict(X_batch)

        for i, (pred, target, sid) in enumerate(zip(preds, targets.numpy(), stock_idx.numpy())):
            stock_data[sid]["preds"].append(pred)
            stock_data[sid]["targets"].append(target)

        del X_batch

    # Compute IC per stock
    test_ic_per_stock = []
    for stock_idx in range(len(symbols)):
        preds = stock_data[stock_idx]["preds"]
        targets = stock_data[stock_idx]["targets"]

        if len(preds) < 2:
            test_ic_per_stock.append(0.0)
            continue

        ic, _ = spearmanr(preds, targets)
        if np.isnan(ic) or ic is None:
            ic = 0.0
        test_ic_per_stock.append(ic)

    test_ic_per_stock = np.array(test_ic_per_stock)
    test_avg_ic = np.mean(np.abs(test_ic_per_stock))

    del stock_data
    gc.collect()

    logger.info(f"Test IC: {test_avg_ic:.4f}")
    logger.info(f"Test IC per stock (top 10): {dict(list(zip(symbols, test_ic_per_stock))[:10])}")

    return {
        "model": "Linear",
        "train_ic": train_avg_ic,
        "test_ic": test_avg_ic,
        "test_ic_per_stock": test_ic_per_stock,
        "normalization": {
            "mean": running_mean,
            "std": running_std,
        },  # Save for later use
    }


def train_transformer_model(
    train_dataset,
    test_dataset,
    symbols,
    epochs=20,
    lr=1e-3,
    batch_size=64,
    gradient_accumulation_steps=4,
):
    """Train a simple transformer model with memory-efficient settings.

    Args:
        gradient_accumulation_steps: Accumulate gradients over N steps to simulate larger batch size
                                     Effective batch size = batch_size * gradient_accumulation_steps
    """
    logger.info("Training Transformer Model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

    # Create data loaders with reduced batch size and no pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Initialize model
    # Use enhanced_num_features since technical features are computed on-the-fly
    input_dim = train_dataset.enhanced_num_features
    logger.info(f"Model input dimension: {input_dim} (enhanced features per timestep)")

    model = SimpleTransformer(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Enable mixed precision training for memory efficiency
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using automatic mixed precision (AMP) training")

    # Training loop with tracking and gradient accumulation
    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (features, targets, _, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            features = features.to(device, non_blocking=False)
            targets = targets.to(device, non_blocking=False)

            # Use automatic mixed precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    predictions = model(features, train_dataset.window_size)
                    loss = criterion(predictions, targets)
                    loss = loss / gradient_accumulation_steps  # Normalize loss

                scaler.scale(loss).backward()
            else:
                predictions = model(features, train_dataset.window_size)
                loss = criterion(predictions, targets)
                loss = loss / gradient_accumulation_steps  # Normalize loss
                loss.backward()

            # Update weights every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Clear GPU cache periodically
                if device.type == "cuda" and (batch_idx + 1) % (gradient_accumulation_steps * 10) == 0:
                    torch.cuda.empty_cache()

            train_loss += loss.item() * gradient_accumulation_steps  # Denormalize for logging

            # Clean up
            del features, targets, predictions, loss

        # Handle any remaining gradients
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        scheduler.step()

        # Force garbage collection
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker="o", linewidth=2, markersize=6)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Transformer Model Training Loss", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"transformer_training_loss_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Training loss plot saved to {plot_path}")
    plt.close()

    # Evaluate on training set STREAMING
    model.eval()
    logger.info("Evaluating on train set (streaming)...")

    stock_data = {i: {"preds": [], "targets": []} for i in range(len(symbols))}

    with torch.no_grad():
        for features, targets, stock_idx, _ in tqdm(train_loader, desc="Evaluating on train set"):
            features = features.to(device, non_blocking=False)

            # Use mixed precision for inference too
            if use_amp:
                with torch.cuda.amp.autocast():
                    predictions = model(features, train_dataset.window_size)
            else:
                predictions = model(features, train_dataset.window_size)

            # Store per-stock instead of concatenating all
            preds_np = predictions.cpu().numpy()
            targets_np = targets.numpy()
            stock_idx_np = stock_idx.numpy()

            for pred, target, sid in zip(preds_np, targets_np, stock_idx_np):
                stock_data[sid]["preds"].append(pred)
                stock_data[sid]["targets"].append(target)

            # Clean up
            del features, predictions, preds_np, targets_np, stock_idx_np

    # Compute IC per stock
    train_ic_per_stock = []
    for stock_idx in range(len(symbols)):
        preds = stock_data[stock_idx]["preds"]
        targets = stock_data[stock_idx]["targets"]

        if len(preds) < 2:
            train_ic_per_stock.append(0.0)
            continue

        ic, _ = spearmanr(preds, targets)
        if np.isnan(ic) or ic is None:
            ic = 0.0
        train_ic_per_stock.append(ic)

    train_ic_per_stock = np.array(train_ic_per_stock)
    train_avg_ic = np.mean(np.abs(train_ic_per_stock))

    del stock_data

    # Clear cache after evaluation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Train IC: {train_avg_ic:.4f}")

    # Evaluate on test set STREAMING
    logger.info("Evaluating on test set (streaming)...")
    stock_data = {i: {"preds": [], "targets": []} for i in range(len(symbols))}

    with torch.no_grad():
        for features, targets, stock_idx, _ in tqdm(test_loader, desc="Evaluating on test set"):
            features = features.to(device, non_blocking=False)

            # Use mixed precision for inference too
            if use_amp:
                with torch.cuda.amp.autocast():
                    predictions = model(features, train_dataset.window_size)
            else:
                predictions = model(features, train_dataset.window_size)

            preds_np = predictions.cpu().numpy()
            targets_np = targets.numpy()
            stock_idx_np = stock_idx.numpy()

            for pred, target, sid in zip(preds_np, targets_np, stock_idx_np):
                stock_data[sid]["preds"].append(pred)
                stock_data[sid]["targets"].append(target)

            # Clean up
            del features, predictions, preds_np, targets_np, stock_idx_np

    # Compute IC per stock
    test_ic_per_stock = []
    for stock_idx in range(len(symbols)):
        preds = stock_data[stock_idx]["preds"]
        targets = stock_data[stock_idx]["targets"]

        if len(preds) < 2:
            test_ic_per_stock.append(0.0)
            continue

        ic, _ = spearmanr(preds, targets)
        if np.isnan(ic) or ic is None:
            ic = 0.0
        test_ic_per_stock.append(ic)

    test_ic_per_stock = np.array(test_ic_per_stock)
    test_avg_ic = np.mean(np.abs(test_ic_per_stock))

    del stock_data

    # Clear cache after evaluation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    logger.info(f"Test IC: {test_avg_ic:.4f}")
    logger.info(f"Test IC per stock (top 10): {dict(list(zip(symbols, test_ic_per_stock))[:10])}")

    return {
        "model": "Transformer",
        "train_ic": train_avg_ic,
        "test_ic": test_avg_ic,
        "test_ic_per_stock": test_ic_per_stock,
    }


def main():
    parser = argparse.ArgumentParser(description="Train supervised baseline models for financial return prediction")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="List of stock tickers to use (default: all available tickers from NYSE and NASDAQ)",
    )
    parser.add_argument("--train-start", type=str, default="2020-01-01", help="Training start date")
    parser.add_argument("--train-end", type=str, default="2023-12-31", help="Training end date")
    parser.add_argument(
        "--test-start",
        type=str,
        default="2024-01-02",
        help="Test start date (1 day gap)",
    )
    parser.add_argument("--test-end", type=str, default="2024-06-30", help="Test end date")
    parser.add_argument("--window-size", type=int, default=30, help="Historical window size (minutes)")
    parser.add_argument(
        "--prediction-horizon",
        type=int,
        default=30,
        help="Prediction horizon (minutes)",
    )
    parser.add_argument(
        "--time-sampling-rate",
        type=int,
        default=5,
        help="Sample every Nth minute (1=all, 5=every 5min, reduces dataset by 5×)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "transformer", "both"],
        default="both",
        help="Which model to train",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for transformer")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for transformer (reduced for memory efficiency)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * this)",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for transformer")

    args = parser.parse_args()

    # If no tickers specified, use all available tickers
    if args.tickers is None:
        args.tickers = get_all_tickers()
        logger.info(f"Using all available tickers: {len(args.tickers)} tickers")

    logger.info("=" * 80)
    logger.info("Supervised Baseline Training for Financial Return Prediction")
    logger.info("=" * 80)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Train period: {args.train_start} to {args.train_end}")
    logger.info(f"Test period: {args.test_start} to {args.test_end}")
    logger.info(f"Window size: {args.window_size} minutes")
    logger.info(f"Prediction horizon: {args.prediction_horizon} minutes")
    logger.info("=" * 80)

    # Use swm.World to manage data loading through the framework
    logger.info("Initializing World for data collection...")

    # For supervised learning, we still use load_market_data directly to get
    # multi-stock data in the right format (T, num_stocks, features)
    # The World/environments are designed for single-stock trading episodes
    logger.info("Loading training data...")
    train_data = load_market_data(
        tickers=args.tickers,
        start_time=args.train_start,
        end_time=args.train_end,
        freq="1min",
    )

    # Load test data
    logger.info("Loading test data...")
    test_data = load_market_data(
        tickers=args.tickers,
        start_time=args.test_start,
        end_time=args.test_end,
        freq="1min",
    )

    # CRITICAL: Force float32 immediately (cuts memory in half)
    logger.info("Converting to float32...")
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    logger.info(f"Train data shape: {train_data.shape}, dtype: {train_data.dtype}")
    logger.info(f"Test data shape: {test_data.shape}, dtype: {test_data.dtype}")

    # DO NOT pre-compute technical features (memory killer!)
    # Features will be computed on-the-fly in the dataset

    # Create datasets (pass RAW data only - tech features computed on-the-fly)
    logger.info("Creating datasets...")
    logger.info(f"Time sampling rate: {args.time_sampling_rate}× (reduces dataset size)")

    train_dataset = FinancialDataset(
        train_data,  # RAW data only!
        args.tickers,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        time_sampling_rate=args.time_sampling_rate,
    )

    test_dataset = FinancialDataset(
        test_data,  # RAW data only!
        args.tickers,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
        time_sampling_rate=args.time_sampling_rate,
    )

    # Clear original data if we don't need it anymore
    del train_data, test_data
    gc.collect()
    logger.info("Original data arrays cleared from memory")

    # Train models
    results = []

    if args.model in ["linear", "both"]:
        linear_results = train_linear_model(train_dataset, test_dataset, args.tickers)
        results.append(linear_results)

    if args.model in ["transformer", "both"]:
        transformer_results = train_transformer_model(
            train_dataset,
            test_dataset,
            args.tickers,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        results.append(transformer_results)

        # Clean up after transformer training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    for result in results:
        logger.info(f"\n{result['model']} Model:")
        logger.info(f"  Train IC: {result['train_ic']:.4f}")
        logger.info(f"  Test IC:  {result['test_ic']:.4f}")

    # Create comparison plot if both models were trained
    if len(results) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot IC comparison
        models = [r["model"] for r in results]
        train_ics = [r["train_ic"] for r in results]
        test_ics = [r["test_ic"] for r in results]

        x = np.arange(len(models))
        width = 0.35

        ax1.bar(x - width / 2, train_ics, width, label="Train IC", alpha=0.8)
        ax1.bar(x + width / 2, test_ics, width, label="Test IC", alpha=0.8)
        ax1.set_xlabel("Model", fontsize=12)
        ax1.set_ylabel("Information Coefficient", fontsize=12)
        ax1.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot top stocks by IC for each model (up to 10)
        num_stocks_to_plot = min(10, len(args.tickers))

        for idx, result in enumerate(results):
            ic_per_stock = result["test_ic_per_stock"]
            top_indices = np.argsort(ic_per_stock)[-num_stocks_to_plot:][::-1]
            _top_tickers = [args.tickers[i] for i in top_indices]
            top_ics = [ic_per_stock[i] for i in top_indices]

            offset = idx * 0.4 - 0.2
            _bars = ax2.barh(
                [i + offset for i in range(num_stocks_to_plot)],
                top_ics,
                height=0.35,
                label=result["model"],
                alpha=0.8,
            )

        ax2.set_yticks(range(num_stocks_to_plot))
        ax2.set_yticklabels(
            [args.tickers[i] for i in np.argsort(results[0]["test_ic_per_stock"])[-num_stocks_to_plot:][::-1]]
        )
        ax2.set_xlabel("Test IC", fontsize=12)
        ax2.set_title(
            f"Top {num_stocks_to_plot} Stocks by Test IC",
            fontsize=14,
            fontweight="bold",
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"model_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"\nComparison plot saved to {plot_path}")
        plt.close()


if __name__ == "__main__":
    main()
