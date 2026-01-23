"""I-JEPA style world model for financial trading environment.

Implements Joint-Embedding Predictive Architecture (I-JEPA) as a world model
that learns to predict masked next-state features from (state, action) context.

Uses swm.World("swm/Financial-v0") to collect trajectories and learn environment dynamics.

Key features:
- Collects real environment transitions using swm.World
- Context encoder processes (state, action) pairs
- Target encoder processes next_state (momentum updated)
- Predictor learns to reconstruct masked next_state representations
- True world model: predicts next observation given current state + action

Usage:
    # Quick test
    python scripts/train/ijepa_worldmodel.py --tickers AAPL \\
        --train-start 2023-01-03 --train-end 2023-01-05 \\
        --episodes 10 --steps-per-episode 50 --epochs 5

    # Full training
    python scripts/train/ijepa_worldmodel.py --tickers AAPL MSFT GOOGL \\
        --train-start 2022-01-03 --train-end 2023-12-31 \\
        --episodes 500 --steps-per-episode 100 --epochs 50 --num-envs 4
"""

import argparse
import gc
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import stable_worldmodel as swm


# Suppress numpy casting warnings from financial data processing
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")


def collect_trajectories(world, num_episodes, steps_per_episode, seed=42):
    """Collect trajectories from financial trading environment.

    Args:
        seed: Random seed for episode resets

    Returns:
        List of (state, action, next_state, reward, done) transitions
    """
    transitions = []

    logger.info(f"Collecting {num_episodes} episodes x {steps_per_episode} steps...")

    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        world.reset(seed=seed + episode)
        states = world.states

        for step in range(steps_per_episode):
            actions = world.policy.get_action(states)
            next_states, rewards, terminated, truncated, infos = world.envs.step(actions)

            # Store transitions from each parallel environment
            for env_idx in range(world.num_envs):
                transitions.append(
                    {
                        "state": states[env_idx].copy(),
                        "action": actions[env_idx].copy(),
                        "next_state": next_states[env_idx].copy(),
                        "reward": rewards[env_idx],
                        "done": terminated[env_idx] or truncated[env_idx],
                    }
                )

            states = next_states

            if all(terminated) or all(truncated):
                break

    logger.info(f"Collected {len(transitions)} transitions")
    return transitions


class WorldModelDataset(Dataset):
    """Dataset for world model learning with I-JEPA style masking.

    Memory optimized: lazy loading, no pre-computed windows.
    """

    def __init__(self, transitions, mask_ratio=0.3):
        self.transitions = transitions
        self.mask_ratio = mask_ratio

        # Get dimensions
        self.state_dim = transitions[0]["state"].shape[0]
        # Action might be a scalar or an array
        action = transitions[0]["action"]
        self.action_dim = action.shape[0] if hasattr(action, "shape") and len(action.shape) > 0 else 1

        logger.info(
            f"Dataset: {len(transitions)} transitions, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"mask_ratio={mask_ratio} (lazy loading enabled)"
        )

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        state = t["state"]
        action = t["action"]
        next_state = t["next_state"]

        # Ensure action is a 1D array (even if it's a scalar)
        if not hasattr(action, "shape") or len(action.shape) == 0:
            action = np.array([action], dtype=np.float32)
        elif len(action.shape) == 1 and action.shape[0] == 1:
            action = action.astype(np.float32)
        else:
            action = action.flatten().astype(np.float32)

        # Create random feature mask for next_state
        num_masked = int(self.state_dim * self.mask_ratio)
        mask = np.zeros(self.state_dim, dtype=np.float32)
        if num_masked > 0:
            masked_idx = np.random.choice(self.state_dim, num_masked, replace=False)
            mask[masked_idx] = 1.0

        # Apply mask
        masked_next_state = next_state.copy()
        masked_next_state[mask > 0] = 0.0

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor(masked_next_state, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(t["reward"], dtype=torch.float32),
        )


class StateEncoder(nn.Module):
    """Encodes state observations (used for both context and target in I-JEPA)."""

    def __init__(self, state_dim, d_model=128, num_layers=3, dropout=0.1, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        layers = [
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.encoder = nn.Sequential(*layers)

    def forward(self, state):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.encoder, state, use_reentrant=False)
        return self.encoder(state)


class NextStatePredictor(nn.Module):
    """Predicts next_state representation from context state representation + action."""

    def __init__(self, d_model=128, action_dim=1, num_layers=2):
        super().__init__()

        # Project action to same dimension as state representation
        self.action_proj = nn.Linear(action_dim, d_model)

        layers = []
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
        self.predictor = nn.Sequential(*layers)

    def forward(self, context_emb, action):
        # Combine context representation with action
        action_emb = self.action_proj(action)
        combined = context_emb + action_emb  # Additive combination
        return self.predictor(combined)


class ReturnPredictionHead(nn.Module):
    """Predicts 30-minute returns from world model representations."""

    def __init__(self, d_model=128, num_stocks=1):
        super().__init__()

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_stocks),
        )

    def forward(self, repr_emb):
        """Predict returns from representation.

        Args:
            repr_emb: (batch, d_model) representation from world model

        Returns:
            returns: (batch, num_stocks) predicted 30-min returns
        """
        return self.prediction_head(repr_emb)


class IJEPAWorldModel(nn.Module):
    """I-JEPA world model following the original paper architecture.

    Key insight: Both encoders have the same architecture and process states.
    The difference is context encoder processes current state, target encoder
    processes next state. Predictor combines context + action to predict next.

    Memory optimized with gradient checkpointing.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        d_model=128,
        num_layers=3,
        dropout=0.1,
        momentum=0.996,
        use_checkpointing=True,
    ):
        super().__init__()

        self.momentum = momentum

        # Both encoders have same architecture (I-JEPA principle)
        self.context_encoder = StateEncoder(state_dim, d_model, num_layers, dropout, use_checkpointing)
        self.target_encoder = StateEncoder(state_dim, d_model, num_layers, dropout, use_checkpointing)
        self.predictor = NextStatePredictor(d_model, action_dim, num_layers=2)

        # Initialize target encoder with same weights as context encoder
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        # Freeze target encoder (updated via momentum only)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """Momentum update of target encoder (EMA of context encoder)."""
        for param_c, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_c.data

    def forward(self, state, action, next_state, masked_next_state, mask):
        # Encode current state (context)
        context_emb = self.context_encoder(state)

        # Predict next_state representation from context + action
        pred_next_repr = self.predictor(context_emb, action)

        # Target next_state representation (no grad, from momentum encoder)
        with torch.no_grad():
            target_next_repr = self.target_encoder(next_state)

        return pred_next_repr, target_next_repr


def train_world_model(train_dataset, args):
    """Train I-JEPA world model with memory optimizations."""
    logger.info("Training I-JEPA World Model (Memory Optimized)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Enable memory optimizations
    if torch.cuda.is_available():
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    # Reduce effective batch size with gradient accumulation
    effective_batch_size = args.batch_size
    actual_batch_size = max(1, args.batch_size // args.grad_accum_steps)
    logger.info(
        f"Effective batch: {effective_batch_size}, Actual batch: {actual_batch_size}, Accum steps: {args.grad_accum_steps}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    model = IJEPAWorldModel(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        d_model=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        momentum=args.momentum,
        use_checkpointing=args.use_checkpointing,
    ).to(device)

    # Only train context encoder and predictor (target encoder is momentum-updated)
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision training
    scaler = GradScaler(enabled=args.use_amp)

    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (
            state,
            action,
            next_state,
            masked_next_state,
            mask,
            reward,
        ) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            state, action, next_state = (
                state.to(device, non_blocking=True),
                action.to(device, non_blocking=True),
                next_state.to(device, non_blocking=True),
            )
            masked_next_state, mask = (
                masked_next_state.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
            )

            # Mixed precision forward pass
            with autocast(enabled=args.use_amp):
                pred_repr, target_repr = model(state, action, next_state, masked_next_state, mask)
                loss = F.mse_loss(pred_repr, target_repr.detach())
                loss = loss / args.grad_accum_steps

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            # Update weights after accumulation steps
            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                model.update_target_encoder()

            epoch_loss += loss.item() * args.grad_accum_steps

            # Periodic memory cleanup
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        scheduler.step()

        # Memory cleanup after epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        if torch.cuda.is_available():
            logger.info(
                f"CUDA Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB / {torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak"
            )

    # Save
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("I-JEPA World Model Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"ijepa_worldmodel_loss_{timestamp}.png", dpi=150)
    logger.info(f"Plot saved: ijepa_worldmodel_loss_{timestamp}.png")
    plt.close()

    model_path = f"ijepa_worldmodel_{timestamp}.pt"
    torch.save(
        {
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "state_dim": train_dataset.state_dim,
            "action_dim": train_dataset.action_dim,
            "args": args,
        },
        model_path,
    )
    logger.info(f"Model saved: {model_path}")

    return model


def finetune_return_prediction(world_model, train_world, test_world, args):
    """Fine-tune return prediction head on top of frozen world model."""
    logger.info("\nFine-tuning Return Prediction Head")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = world_model.to(device)
    world_model.eval()  # Freeze world model

    # Create prediction head
    num_stocks = len(args.tickers)
    pred_head = ReturnPredictionHead(d_model=args.hidden_dim, num_stocks=num_stocks).to(device)

    optimizer = torch.optim.Adam(pred_head.parameters(), lr=args.pred_lr)

    # Collect training data for return prediction
    logger.info("Collecting training transitions...")
    train_transitions = collect_trajectories(train_world, args.pred_episodes, args.steps_per_episode, args.seed)

    # Compute 30-minute returns for each transition
    # Financial env observation structure: [price_window (window_size*6), portfolio (3), time (4)]
    # Price window has 6 features per timestep: [open, high, low, close, volume, vwap]
    logger.info("Computing target returns...")
    window_size = (
        train_world.envs.envs[0].unwrapped.window_size if hasattr(train_world.envs.envs[0], "unwrapped") else 30
    )

    for t in train_transitions:
        # Extract normalized close price from most recent timestep in window
        # Close is at position 3 within each 6-feature group
        # Most recent close is at position: (window_size-1) * 6 + 3
        recent_close_idx = (window_size - 1) * 6 + 3

        # Get current and next close prices (these are normalized by current_close in env)
        # So we need to look at the relative change
        current_state_prices = t["state"][: window_size * 6]
        next_state_prices = t["next_state"][: window_size * 6]

        # Get the most recent normalized close from current state
        current_norm_close = current_state_prices[recent_close_idx]
        # Get the most recent normalized close from next state
        next_norm_close = next_state_prices[recent_close_idx]

        # Return is the relative change
        t["return_30min"] = (next_norm_close - current_norm_close) / (current_norm_close + 1e-8)

    # Train prediction head
    train_losses = []

    for epoch in range(args.pred_epochs):
        epoch_loss = 0.0
        np.random.shuffle(train_transitions)

        for i in range(0, len(train_transitions), args.batch_size):
            batch = train_transitions[i : i + args.batch_size]
            if len(batch) < 2:
                continue

            states = torch.stack([torch.tensor(t["state"], dtype=torch.float32) for t in batch]).to(device)
            _actions = torch.stack([torch.tensor(t["action"], dtype=torch.float32) for t in batch]).to(device)
            returns = torch.stack([torch.tensor([t["return_30min"]], dtype=torch.float32) for t in batch]).to(device)

            optimizer.zero_grad()

            # Get frozen representations from world model (just state, not action)
            with torch.no_grad():
                repr_emb = world_model.context_encoder(states)

            # Predict returns
            pred_returns = pred_head(repr_emb)

            loss = F.mse_loss(pred_returns, returns)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_transitions) // args.batch_size)
        train_losses.append(epoch_loss)

        if (epoch + 1) % 5 == 0:
            logger.info(f"Prediction Epoch {epoch + 1}/{args.pred_epochs}, Loss: {epoch_loss:.6f}")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_transitions = collect_trajectories(test_world, args.pred_episodes // 2, args.steps_per_episode, args.seed)

    test_window_size = (
        test_world.envs.envs[0].unwrapped.window_size if hasattr(test_world.envs.envs[0], "unwrapped") else 30
    )
    for t in test_transitions:
        recent_close_idx = (test_window_size - 1) * 6 + 3
        current_state_prices = t["state"][: test_window_size * 6]
        next_state_prices = t["next_state"][: test_window_size * 6]
        current_norm_close = current_state_prices[recent_close_idx]
        next_norm_close = next_state_prices[recent_close_idx]
        t["return_30min"] = (next_norm_close - current_norm_close) / (current_norm_close + 1e-8)

    pred_head.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(test_transitions), args.batch_size):
            batch = test_transitions[i : i + args.batch_size]
            if len(batch) < 1:
                continue

            states = torch.stack([torch.tensor(t["state"], dtype=torch.float32) for t in batch]).to(device)
            _actions = torch.stack([torch.tensor(t["action"], dtype=torch.float32) for t in batch]).to(device)
            returns = torch.stack([torch.tensor([t["return_30min"]], dtype=torch.float32) for t in batch]).to(device)

            repr_emb = world_model.context_encoder(states)
            pred_returns = pred_head(repr_emb)

            all_preds.append(pred_returns.cpu().numpy())
            all_targets.append(returns.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute Information Coefficient (Spearman correlation)
    ic_values = []
    for stock_idx in range(num_stocks):
        if all_preds.shape[1] > stock_idx:
            ic, _ = spearmanr(all_preds[:, stock_idx], all_targets[:, stock_idx])
            ic_values.append(ic)
            logger.info(f"Stock {args.tickers[stock_idx]}: IC = {ic:.4f}")

    mean_ic = np.mean(ic_values)
    logger.info(f"\nMean Information Coefficient: {mean_ic:.4f}")
    logger.info("=" * 80)

    # Save prediction head
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_head_path = f"ijepa_pred_head_{timestamp}.pt"
    torch.save(
        {
            "pred_head": pred_head.state_dict(),
            "mean_ic": mean_ic,
            "ic_values": ic_values,
            "args": args,
        },
        pred_head_path,
    )
    logger.info(f"Prediction head saved: {pred_head_path}")

    return pred_head, mean_ic


def main():
    parser = argparse.ArgumentParser(description="Train I-JEPA world model")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="Stock tickers")
    parser.add_argument("--train-start", type=str, default="2023-01-03")
    parser.add_argument("--train-end", type=str, default="2023-01-05")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel environments")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes to collect")
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Effective batch size (reduced for memory)",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable AMP")
    parser.add_argument(
        "--use-checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--no-checkpointing",
        dest="use_checkpointing",
        action="store_false",
        help="Disable checkpointing",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Prediction head arguments
    parser.add_argument("--test-start", type=str, default="2023-01-06", help="Test start date")
    parser.add_argument("--test-end", type=str, default="2023-01-10", help="Test end date")
    parser.add_argument("--pred-epochs", type=int, default=20, help="Epochs for prediction head")
    parser.add_argument("--pred-episodes", type=int, default=50, help="Episodes for prediction training")
    parser.add_argument("--pred-lr", type=float, default=1e-3, help="Learning rate for prediction head")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip prediction head training")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    import random

    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 80)
    logger.info("I-JEPA World Model Training for Financial Trading")
    logger.info("=" * 80)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Period: {args.train_start} to {args.train_end}")
    logger.info(f"Envs: {args.num_envs}, Episodes: {args.episodes}, Steps: {args.steps_per_episode}")
    logger.info(f"Mask ratio: {args.mask_ratio}")
    logger.info("=" * 80)

    # Create world
    logger.info("Creating world...")
    world = swm.World(
        "swm/Financial-v0",
        num_envs=args.num_envs,
        render_mode=None,
        symbols=args.tickers,
        start_date=args.train_start,
        end_date=args.train_end,
    )

    # Set random policy
    world.set_policy(swm.policy.RandomPolicy())

    # Collect trajectories
    transitions = collect_trajectories(world, args.episodes, args.steps_per_episode, args.seed)

    # Create dataset
    dataset = WorldModelDataset(transitions, mask_ratio=args.mask_ratio)

    # Train world model
    model = train_world_model(dataset, args)

    # Fine-tune prediction head for comparison with supervised baseline
    if not args.skip_prediction:
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Fine-tuning Return Prediction Head")
        logger.info("=" * 80)

        # Create test world
        test_world = swm.World(
            "swm/Financial-v0",
            num_envs=args.num_envs,
            render_mode=None,
            symbols=args.tickers,
            start_date=args.test_start,
            end_date=args.test_end,
        )
        test_world.set_policy(swm.policy.RandomPolicy())

        pred_head, mean_ic = finetune_return_prediction(model, world, test_world, args)

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("World model learned to predict next_state from (state, action)")
    if not args.skip_prediction:
        logger.info(f"Return prediction IC: {mean_ic:.4f}")
        logger.info("Can compare with supervised baseline using IC metric")
    logger.info("Can be used for: planning, model-based RL, simulation")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
