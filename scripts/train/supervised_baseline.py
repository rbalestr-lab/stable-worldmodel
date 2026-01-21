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

    # Train transformer model with custom hyperparameters
    python scripts/train/supervised_baseline.py --model transformer --tickers AAPL MSFT \\
        --train-start 2022-01-03 --train-end 2023-12-31 \\
        --test-start 2024-01-02 --test-end 2024-03-31 \\
        --epochs 50 --batch-size 64 --lr 1e-4
"""

import argparse
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
from sklearn.preprocessing import StandardScaler
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
    """PyTorch Dataset wrapper for financial time series prediction.

    Wraps data loaded from stable_worldmodel's finance data module.
    """

    def __init__(
        self,
        data: np.ndarray,  # Shape: (T, num_stocks, features)
        symbols: list[str],
        window_size: int = 30,
        prediction_horizon: int = 30,
    ):
        """Initialize financial dataset.

        Args:
            data: Market data array (T, num_stocks, features) - should already include technical features
            symbols: List of stock symbols
            window_size: Number of historical minutes to use
            prediction_horizon: Minutes ahead to predict returns
        """
        self.data = data
        self.symbols = symbols
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.num_stocks = data.shape[1]
        self.num_features = data.shape[2]

        # Compute returns (30-minute ahead)
        close_prices = data[:, :, 3]  # Close is 4th feature
        future_close = np.roll(close_prices, -prediction_horizon, axis=0)
        self.returns = (future_close - close_prices) / (close_prices + 1e-8)

        # Valid indices (have enough history and future)
        self.valid_indices = list(range(window_size, len(data) - prediction_horizon))

        logger.info(
            f"Dataset initialized: {len(self.valid_indices)} samples, "
            f"{self.num_stocks} stocks, {self.num_features} features"
        )

    def __len__(self):
        return len(self.valid_indices) * self.num_stocks

    def __getitem__(self, idx):
        """Get a single sample.

        Returns:
            features: (window_size * num_features,) flattened historical features
            target: scalar future return
            stock_idx: which stock this sample is for
            time_idx: which time this sample is for
        """
        stock_idx = idx % self.num_stocks
        time_idx_in_valid = idx // self.num_stocks
        time_idx = self.valid_indices[time_idx_in_valid]

        # Get historical window
        start_t = time_idx - self.window_size
        historical_features = self.data[start_t:time_idx, stock_idx, :]  # (window, features)

        # Flatten to 1D
        features = historical_features.flatten()

        # Get target return
        target = self.returns[time_idx, stock_idx]

        return (
            torch.tensor(features, dtype=torch.float32),
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


def train_linear_model(train_dataset, test_dataset, symbols, batch_size=1024):
    """Train a linear (SGD) regression model with mini-batch training."""
    logger.info("Training Linear Model (SGDRegressor)")

    # Create data loader for mini-batch training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize StandardScaler - fit incrementally
    scaler = StandardScaler()
    logger.info("Fitting scaler on training data...")
    for features, _, _, _ in tqdm(train_loader, desc="Fitting scaler"):
        scaler.partial_fit(features.numpy())

    # Initialize SGDRegressor
    model = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        learning_rate="adaptive",
        eta0=0.01,
        max_iter=1,
        tol=None,
        random_state=42,
    )

    # Train in mini-batches and track losses
    n_epochs = 5
    epoch_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for features, targets, _, _ in tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{n_epochs}"):
            X_batch = scaler.transform(features.numpy())
            y_batch = targets.numpy()
            model.partial_fit(X_batch, y_batch)

            # Compute batch loss
            preds = model.predict(X_batch)
            loss = np.mean((preds - y_batch) ** 2)
            epoch_loss += loss
            n_batches += 1

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

    # Evaluate on training set (in batches to avoid memory issues)
    logger.info("Evaluating on training set...")
    train_preds = []
    train_targets = []
    train_stock_indices = []

    for features, targets, stock_idx, _ in tqdm(train_loader, desc="Train evaluation"):
        X_batch = scaler.transform(features.numpy())
        preds = model.predict(X_batch)
        train_preds.append(preds)
        train_targets.append(targets.numpy())
        train_stock_indices.append(stock_idx.numpy())

    train_preds = np.concatenate(train_preds)
    train_targets = np.concatenate(train_targets)
    train_stock_indices = np.concatenate(train_stock_indices)

    train_ic_per_stock, train_avg_ic = compute_ic_per_stock(
        train_preds, train_targets, train_stock_indices, len(symbols)
    )
    logger.info(f"Train IC: {train_avg_ic:.4f}")

    # Evaluate on test set (in batches)
    logger.info("Evaluating on test set...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_preds = []
    test_targets = []
    test_stock_indices = []

    for features, targets, stock_idx, _ in tqdm(test_loader, desc="Test evaluation"):
        X_batch = scaler.transform(features.numpy())
        preds = model.predict(X_batch)
        test_preds.append(preds)
        test_targets.append(targets.numpy())
        test_stock_indices.append(stock_idx.numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_stock_indices = np.concatenate(test_stock_indices)

    test_ic_per_stock, test_avg_ic = compute_ic_per_stock(test_preds, test_targets, test_stock_indices, len(symbols))

    logger.info(f"Test IC: {test_avg_ic:.4f}")
    logger.info(f"Test IC per stock:\n{dict(zip(symbols, test_ic_per_stock))}")

    return {
        "model": "Linear",
        "train_ic": train_avg_ic,
        "test_ic": test_avg_ic,
        "test_ic_per_stock": test_ic_per_stock,
    }


def train_transformer_model(train_dataset, test_dataset, symbols, epochs=20, lr=1e-3, batch_size=256):
    """Train a simple transformer model."""
    logger.info("Training Transformer Model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    input_dim = train_dataset.num_features
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

    # Training loop with tracking
    train_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features, targets, _, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features, train_dataset.window_size)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        scheduler.step()

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

    # Evaluate on training set
    model.eval()
    train_preds = []
    train_targets = []
    train_stock_indices = []

    with torch.no_grad():
        for features, targets, stock_idx, _ in tqdm(train_loader, desc="Evaluating on train set"):
            features = features.to(device)
            predictions = model(features, train_dataset.window_size)
            train_preds.append(predictions.cpu().numpy())
            train_targets.append(targets.numpy())
            train_stock_indices.append(stock_idx.numpy())

    train_preds = np.concatenate(train_preds)
    train_targets = np.concatenate(train_targets)
    train_stock_indices = np.concatenate(train_stock_indices)

    train_ic_per_stock, train_avg_ic = compute_ic_per_stock(
        train_preds, train_targets, train_stock_indices, len(symbols)
    )
    logger.info(f"Train IC: {train_avg_ic:.4f}")

    # Evaluate on test set
    test_preds = []
    test_targets = []
    test_stock_indices = []

    with torch.no_grad():
        for features, targets, stock_idx, _ in tqdm(test_loader, desc="Evaluating on test set"):
            features = features.to(device)
            predictions = model(features, test_dataset.window_size)
            test_preds.append(predictions.cpu().numpy())
            test_targets.append(targets.numpy())
            test_stock_indices.append(stock_idx.numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_stock_indices = np.concatenate(test_stock_indices)

    test_ic_per_stock, test_avg_ic = compute_ic_per_stock(test_preds, test_targets, test_stock_indices, len(symbols))

    logger.info(f"Test IC: {test_avg_ic:.4f}")
    logger.info(f"Test IC per stock:\n{dict(zip(symbols, test_ic_per_stock))}")

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
        "--model",
        type=str,
        choices=["linear", "transformer", "both"],
        default="both",
        help="Which model to train",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for transformer")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for transformer")
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

    # Add technical features to the data
    logger.info("Adding technical features...")
    train_data_enhanced = add_technical_features(train_data)
    test_data_enhanced = add_technical_features(test_data)

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = FinancialDataset(
        train_data_enhanced,
        args.tickers,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
    )

    test_dataset = FinancialDataset(
        test_data_enhanced,
        args.tickers,
        window_size=args.window_size,
        prediction_horizon=args.prediction_horizon,
    )

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
        )
        results.append(transformer_results)

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
