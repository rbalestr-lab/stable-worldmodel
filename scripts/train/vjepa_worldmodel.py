"""V-JEPA style world model for financial trading environment.

Implements Video Joint-Embedding Predictive Architecture (V-JEPA) as a world model
that learns to predict future state representations given state-action history.

Uses swm.World("swm/Financial-v0") to collect trajectories and learn temporal dynamics.

Key features:
- Collects real environment transitions using swm.World
- Context encoder processes historical (state, action) sequences
- Target encoder processes full temporal sequence (momentum updated)
- Predictor learns to forecast future state representations
- True world model: predicts future observations from history + actions

Usage:
    # Quick test
    python scripts/train/vjepa_worldmodel.py --tickers AAPL \\
        --train-start 2023-01-03 --train-end 2023-01-05 \\
        --episodes 10 --steps-per-episode 50 --epochs 5

    # Full training
    python scripts/train/vjepa_worldmodel.py --tickers AAPL MSFT GOOGL \\
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
        List of full episode trajectories
    """
    trajectories = []

    logger.info(f"Collecting {num_episodes} episodes x {steps_per_episode} steps...")

    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        world.reset(seed=seed + episode)
        states = world.states

        # Store full trajectories for each environment
        for env_idx in range(world.num_envs):
            traj = {
                "states": [states[env_idx].copy()],
                "actions": [],
                "rewards": [],
            }
            trajectories.append(traj)

        for step in range(steps_per_episode):
            actions = world.policy.get_action(states)
            next_states, rewards, terminated, truncated, infos = world.envs.step(actions)

            # Append to each environment's trajectory
            traj_start_idx = len(trajectories) - world.num_envs
            for env_idx in range(world.num_envs):
                traj_idx = traj_start_idx + env_idx
                trajectories[traj_idx]["states"].append(next_states[env_idx].copy())
                # Reshape scalar actions to 1D arrays
                action = actions[env_idx]
                if np.isscalar(action):
                    action = np.array([action])
                trajectories[traj_idx]["actions"].append(action)
                trajectories[traj_idx]["rewards"].append(rewards[env_idx])

            states = next_states

            if all(terminated) or all(truncated):
                break

    logger.info(f"Collected {len(trajectories)} trajectories")
    return trajectories


class TemporalWorldModelDataset(Dataset):
    """Dataset for V-JEPA temporal world model learning.

    Memory optimized: stores trajectory references, creates windows on-the-fly.
    """

    def __init__(self, trajectories, context_len=10, predict_len=5):
        self.trajectories = trajectories
        self.context_len = context_len
        self.predict_len = predict_len

        # Get dimensions
        self.state_dim = trajectories[0]["states"][0].shape[0]
        self.action_dim = trajectories[0]["actions"][0].shape[0]

        # Store indices instead of full windows (memory efficient)
        self.window_indices = []
        for traj_idx, traj in enumerate(trajectories):
            states = traj["states"]
            # Create window indices
            for i in range(len(states) - context_len - predict_len):
                self.window_indices.append((traj_idx, i))

        logger.info(
            f"Dataset: {len(self.window_indices)} windows from {len(trajectories)} trajectories, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"context={context_len}, predict={predict_len} (lazy loading)"
        )

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        traj_idx, start_idx = self.window_indices[idx]
        traj = self.trajectories[traj_idx]

        states = np.array(traj["states"])
        actions = np.array(traj["actions"])

        context_states = states[start_idx : start_idx + self.context_len]
        context_actions = actions[start_idx : start_idx + self.context_len]
        future_states = states[start_idx + self.context_len : start_idx + self.context_len + self.predict_len]

        return (
            torch.tensor(context_states, dtype=torch.float32),
            torch.tensor(context_actions, dtype=torch.float32),
            torch.tensor(future_states, dtype=torch.float32),
        )


class TemporalEncoder(nn.Module):
    """Encodes temporal sequences of (state, action) pairs."""

    def __init__(
        self,
        state_dim,
        action_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        use_checkpointing=False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        input_dim = state_dim + action_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, states, actions):
        """
        Args:
            states: (batch, time, state_dim)
            actions: (batch, time, action_dim)

        Returns:
            embeddings: (batch, time, d_model)
        """
        batch_size, time_steps, _ = states.shape

        # Concatenate states and actions
        x = torch.cat([states, actions], dim=-1)

        # Project
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoder[:, :time_steps, :]

        # Transform with optional checkpointing
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)
        x = self.norm(x)

        return x


class StateEncoder(nn.Module):
    """Encodes state sequences (for target) - NO ACTIONS to avoid leakage."""

    def __init__(
        self,
        state_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        use_checkpointing=False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.input_proj = nn.Linear(state_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, states):
        """
        Args:
            states: (batch, time, state_dim)

        Returns:
            embeddings: (batch, time, d_model)
        """
        batch_size, time_steps, _ = states.shape

        x = self.input_proj(states)
        x = x + self.pos_encoder[:, :time_steps, :]

        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)
        x = self.norm(x)

        return x


class FuturePredictor(nn.Module):
    """Predicts future state representations from context only (NO future actions to avoid leakage)."""

    def __init__(self, d_model=128, predict_len=5, nhead=8, num_layers=2):
        super().__init__()

        self.predict_len = predict_len

        # Learnable queries for future time steps (no action conditioning to avoid leakage)
        self.future_queries = nn.Parameter(torch.randn(1, predict_len, d_model) * 0.02)

        # Cross-attention decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context_emb):
        """
        Args:
            context_emb: (batch, context_time, d_model)

        Returns:
            predictions: (batch, predict_len, d_model)
        """
        batch_size = context_emb.shape[0]

        # Use learnable queries only (no future actions to avoid leakage)
        queries = self.future_queries.expand(batch_size, -1, -1)

        predictions = self.decoder(queries, context_emb)
        predictions = self.norm(predictions)

        return predictions


class ReturnPredictionHead(nn.Module):
    """Prediction head for return forecasting from temporal representations."""

    def __init__(self, d_model=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, temporal_repr):
        """
        Args:
            temporal_repr: (batch, time, d_model)
        Returns:
            returns: (batch, 1) - predicted returns
        """
        # Use last timestep representation
        last_repr = temporal_repr[:, -1, :]
        return self.net(last_repr)


class VJEPAWorldModel(nn.Module):
    """V-JEPA world model for temporal prediction (paper-accurate, no action leakage).

    Key principles from V-JEPA paper:
    - Context encoder: processes historical (state, action) pairs - we know past actions
    - Target encoder: processes future STATES ONLY - avoids action leakage
    - Predictor: predicts future state representations from context (no future actions)
    - Target encoder updated via exponential moving average (momentum)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        d_model=128,
        nhead=8,
        num_layers=4,
        predict_len=5,
        dropout=0.1,
        momentum=0.996,
        use_checkpointing=True,
    ):
        super().__init__()

        self.momentum = momentum
        self.predict_len = predict_len

        # Context encoder: processes historical (state, action) sequences
        self.context_encoder = TemporalEncoder(
            state_dim,
            action_dim,
            d_model,
            nhead,
            num_layers,
            dropout,
            use_checkpointing,
        )

        # Target encoder: processes future STATES only (no actions to avoid leakage)
        self.target_encoder = StateEncoder(state_dim, d_model, nhead, num_layers, dropout, use_checkpointing)

        # Predictor: predicts future from context only (no future actions)
        self.predictor = FuturePredictor(d_model, predict_len, nhead, 2)

        # Target encoder starts with random initialization (different from context encoder)
        # This is correct since they have different architectures (one takes actions, one doesn't)

        # Freeze target encoder - updated only via momentum
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """Momentum update of target encoder (EMA).

        Note: Since context encoder (TemporalEncoder) and target encoder (StateEncoder)
        have different architectures, we only update the shared components:
        - input_proj: Both project to d_model
        - pos_encoder: Both use positional encoding
        - transformer: Both use transformer layers
        - norm: Both use layer norm

        The context encoder's action projection is not copied (target has no actions).
        """
        # Update input projection (context has state+action -> d_model, target has state -> d_model)
        # We update based on the state projection weights from context encoder
        # This provides transfer learning while respecting architectural differences

        # For now, use standard momentum update on matching parameters
        context_dict = dict(self.context_encoder.named_parameters())
        target_dict = dict(self.target_encoder.named_parameters())

        for name_t, param_t in target_dict.items():
            # Find corresponding parameter in context encoder
            # Both have: input_proj, pos_encoder, transformer.*, norm.*
            if name_t in context_dict:
                param_c = context_dict[name_t]
                if param_c.shape == param_t.shape:
                    param_t.data = self.momentum * param_t.data + (1 - self.momentum) * param_c.data

    def forward(self, context_states, context_actions, future_states):
        """
        Args:
            context_states: (batch, context_len, state_dim)
            context_actions: (batch, context_len, action_dim)
            future_states: (batch, predict_len, state_dim)

        Returns:
            pred_future_repr: (batch, predict_len, d_model) - predicted future representations
            target_future_repr: (batch, predict_len, d_model) - target future representations

        Note: No future_actions parameter - this avoids data leakage at test time.
        """
        # Encode context (historical state-action sequence - we know past actions)
        context_emb = self.context_encoder(context_states, context_actions)

        # Predict future representations from context only (no future actions)
        pred_future_repr = self.predictor(context_emb)

        # Target future state representations (no actions, no grad)
        with torch.no_grad():
            target_future_repr = self.target_encoder(future_states)

        return pred_future_repr, target_future_repr


def train_world_model(train_dataset, args):
    """Train V-JEPA world model with memory optimizations."""
    logger.info("Training V-JEPA World Model (Memory Optimized)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Enable memory optimizations
    if torch.cuda.is_available():
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True

    # Gradient accumulation for memory efficiency
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

    model = VJEPAWorldModel(
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        d_model=args.hidden_dim,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        predict_len=args.predict_len,
        dropout=args.dropout,
        momentum=args.momentum,
        use_checkpointing=args.use_checkpointing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (context_states, context_actions, future_states) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):
            context_states = context_states.to(device, non_blocking=True)
            context_actions = context_actions.to(device, non_blocking=True)
            future_states = future_states.to(device, non_blocking=True)

            # Mixed precision forward
            with autocast(enabled=args.use_amp):
                pred_repr, target_repr = model(context_states, context_actions, future_states)
                loss = F.mse_loss(pred_repr, target_repr.detach())
                loss = loss / args.grad_accum_steps

            # Backward with scaling
            scaler.scale(loss).backward()

            # Update after accumulation
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

        # Cleanup after epoch
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
    plt.title("V-JEPA World Model Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"vjepa_worldmodel_loss_{timestamp}.png", dpi=150)
    logger.info(f"Plot saved: vjepa_worldmodel_loss_{timestamp}.png")
    plt.close()

    model_path = f"vjepa_worldmodel_{timestamp}.pt"
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


def finetune_return_prediction(model, world, args, device):
    """Fine-tune return prediction head on frozen world model."""
    from scipy.stats import spearmanr

    logger.info("\n" + "=" * 80)
    logger.info("Fine-tuning Return Prediction Head")
    logger.info("=" * 80)

    # Freeze world model
    for param in model.parameters():
        param.requires_grad = False

    # Create prediction head
    pred_head = ReturnPredictionHead(d_model=args.hidden_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(pred_head.parameters(), lr=1e-3, weight_decay=0.01)

    # Collect training data
    logger.info("Collecting training transitions...")
    train_trajs = collect_trajectories(world, args.pred_episodes, args.steps_per_episode, args.seed)

    # Flatten to transitions and compute returns
    states_list = []
    actions_list = []
    returns_list = []

    logger.info("Computing target returns...")
    for traj in train_trajs:
        states = traj["states"]
        actions = traj["actions"]
        rewards = traj["rewards"]

        # Extract price close from observation
        for i in range(len(states) - 1):
            state = states[i]
            action_seq = actions[max(0, i - args.context_len + 1) : i + 1]

            # Pad action sequence if needed
            if len(action_seq) < args.context_len:
                padding = [np.zeros_like(actions[0]) for _ in range(args.context_len - len(action_seq))]
                action_seq = padding + action_seq

            # Compute return as next step reward
            actual_return = rewards[i]

            states_list.append(state)
            actions_list.append(np.array(action_seq))
            returns_list.append(actual_return)

    states_tensor = torch.tensor(np.array(states_list), dtype=torch.float32).unsqueeze(1).to(device)
    actions_tensor = torch.tensor(np.array(actions_list), dtype=torch.float32).to(device)
    returns_tensor = torch.tensor(np.array(returns_list), dtype=torch.float32).unsqueeze(1).to(device)

    # Train prediction head
    model.eval()
    pred_head.train()

    for epoch in range(args.pred_epochs):
        optimizer.zero_grad()

        # Get world model representations (frozen)
        with torch.no_grad():
            # Expand states to match context length
            batch_states = states_tensor.expand(-1, args.context_len, -1)
            temporal_repr = model.context_encoder(batch_states, actions_tensor)

        # Predict returns
        pred_returns = pred_head(temporal_repr)
        loss = F.mse_loss(pred_returns, returns_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            logger.info(f"Prediction Epoch {epoch + 1}/{args.pred_epochs}, Loss: {loss.item():.6f}")

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_world = swm.World(
        "swm/Financial-v0",
        num_envs=1,
        render_mode=None,
        symbols=args.tickers,
        start_date=args.test_start,
        end_date=args.test_end,
    )
    test_world.set_policy(swm.policy.RandomPolicy())

    test_trajs = collect_trajectories(test_world, 2, args.steps_per_episode, args.seed)

    # Compute IC per stock
    pred_head.eval()
    model.eval()

    test_states = []
    test_actions = []
    test_returns = []

    for traj in test_trajs:
        states = traj["states"]
        actions = traj["actions"]
        rewards = traj["rewards"]

        for i in range(len(states) - 1):
            state = states[i]
            action_seq = actions[max(0, i - args.context_len + 1) : i + 1]

            if len(action_seq) < args.context_len:
                padding = [np.zeros_like(actions[0]) for _ in range(args.context_len - len(action_seq))]
                action_seq = padding + action_seq

            actual_return = rewards[i]

            test_states.append(state)
            test_actions.append(np.array(action_seq))
            test_returns.append(actual_return)

    test_states_tensor = torch.tensor(np.array(test_states), dtype=torch.float32).unsqueeze(1).to(device)
    test_actions_tensor = torch.tensor(np.array(test_actions), dtype=torch.float32).to(device)

    with torch.no_grad():
        batch_states = test_states_tensor.expand(-1, args.context_len, -1)
        temporal_repr = model.context_encoder(batch_states, test_actions_tensor)
        predictions = pred_head(temporal_repr).cpu().numpy().flatten()

    actual = np.array(test_returns)

    # Compute IC
    ic, p_value = spearmanr(predictions, actual)
    logger.info(f"Stock {args.tickers[0]}: IC = {ic:.4f}")
    logger.info(f"\nMean Information Coefficient: {ic:.4f}")
    logger.info("=" * 80)

    # Save prediction head
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_head_path = f"vjepa_pred_head_{timestamp}.pt"
    torch.save(pred_head.state_dict(), pred_head_path)
    logger.info(f"Prediction head saved: {pred_head_path}")

    return ic


def main():
    parser = argparse.ArgumentParser(description="Train V-JEPA world model")
    parser.add_argument("--tickers", nargs="+", default=["AAPL"], help="Stock tickers")
    parser.add_argument("--train-start", type=str, default="2023-01-03")
    parser.add_argument("--train-end", type=str, default="2023-01-05")
    parser.add_argument("--num-envs", type=int, default=2, help="Number of parallel environments")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes to collect")
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--context-len", type=int, default=10, help="Context window length")
    parser.add_argument("--predict-len", type=int, default=5, help="Future steps to predict")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
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
    parser.add_argument("--test-start", type=str, default="2023-01-08", help="Test start date")
    parser.add_argument("--test-end", type=str, default="2023-01-10", help="Test end date")
    parser.add_argument("--pred-epochs", type=int, default=20, help="Prediction head training epochs")
    parser.add_argument(
        "--pred-episodes",
        type=int,
        default=20,
        help="Episodes for prediction head training",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip return prediction head training",
    )

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
    logger.info("V-JEPA World Model Training for Financial Trading")
    logger.info("=" * 80)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Period: {args.train_start} to {args.train_end}")
    logger.info(f"Envs: {args.num_envs}, Episodes: {args.episodes}, Steps: {args.steps_per_episode}")
    logger.info(f"Temporal: context={args.context_len} -> predict={args.predict_len}")
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
    trajectories = collect_trajectories(world, args.episodes, args.steps_per_episode, args.seed)

    # Create dataset
    dataset = TemporalWorldModelDataset(trajectories, context_len=args.context_len, predict_len=args.predict_len)

    # Train world model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_world_model(dataset, args)

    # Train prediction head (unless skipped)
    if not args.skip_prediction:
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Fine-tuning Return Prediction Head")
        logger.info("=" * 80)
        ic = finetune_return_prediction(model, world, args, device)

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("World model learned to predict future states from history + actions")
    if not args.skip_prediction:
        logger.info(f"Return prediction IC: {ic:.4f}")
        logger.info("Can compare with supervised baseline using IC metric")
    logger.info("Can be used for: multi-step planning, forecasting, simulation")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
