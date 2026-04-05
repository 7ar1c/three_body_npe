from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import BoxUniform

Array = np.ndarray
Tensor = torch.Tensor

PARAMETER_NAMES = [
    "x1", "y1", "x2", "y2", "x3", "y3",
    "vx1", "vy1", "vx2", "vy2", "vx3", "vy3",
    "m1", "m2", "m3",
]

@dataclass
class NPEConfig:
    """Configuration for posterior training on precomputed data."""

    num_time_points: int = 32
    num_features: int = 13
    theta_dim: int = 15

    position_min: float = -5.0
    position_max: float = 5.0
    velocity_min: float = -2.0
    velocity_max: float = 2.0
    mass_min: float = 0.5
    mass_max: float = 2.5

    training_batch_size: int = 512
    learning_rate: float = 3e-4
    validation_fraction: float = 0.1
    hidden_features: int = 128
    num_transforms: int = 6
    sequence_embedding_dim: int = 128
    num_posterior_samples: int = 3000

    dataset_path: str = "three_body_dataset.npz"
    plots_dir: str = "./plots"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrajectoryEmbeddingNet(torch.nn.Module):
    """Sequence encoder for observations with shape (batch, 32, 13)."""

    def __init__(self, num_features: int = 13, embedding_dim: int = 128):
        super().__init__()
        self.feature_net = torch.nn.Sequential(
            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
        )
        self.temporal_net = torch.nn.Sequential(
                    torch.nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Dropout1d(p=0.2), # <--- ADD DROPOUT
                    torch.nn.Conv1d(128, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool1d(1),
                )
        self.output_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128, embedding_dim),
            torch.nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (batch, 32, 13), got {tuple(x.shape)}")

        batch_size, time_steps, num_features = x.shape
        if time_steps != 32:
            raise ValueError(f"Expected 32 time steps, got {time_steps}")
        if num_features != 13:
            raise ValueError(f"Expected 13 features per time step, got {num_features}")

        x = x.reshape(batch_size * time_steps, num_features)
        x = self.feature_net(x)
        x = x.reshape(batch_size, time_steps, 64)
        x = x.transpose(1, 2)
        x = self.temporal_net(x)
        x = self.output_net(x)
        return x

def make_prior(cfg: NPEConfig, device: str = "cpu") -> BoxUniform:
    """Create a box prior over the 15 inferred parameters."""
    low = torch.tensor(
        [
            cfg.position_min, cfg.position_min, cfg.position_min, cfg.position_min, cfg.position_min, cfg.position_min,
            cfg.velocity_min, cfg.velocity_min, cfg.velocity_min, cfg.velocity_min, cfg.velocity_min, cfg.velocity_min,
            cfg.mass_min, cfg.mass_min, cfg.mass_min,
        ],
        dtype=torch.float32,
        device=device 
    )
    high = torch.tensor(
        [
            cfg.position_max, cfg.position_max, cfg.position_max, cfg.position_max, cfg.position_max, cfg.position_max,
            cfg.velocity_max, cfg.velocity_max, cfg.velocity_max, cfg.velocity_max, cfg.velocity_max, cfg.velocity_max,
            cfg.mass_max, cfg.mass_max, cfg.mass_max,
        ],
        dtype=torch.float32,
        device=device
    )
    return BoxUniform(low=low, high=high)

def validate_dataset(theta: Array, x: Array, cfg: NPEConfig) -> None:
    """Validate dataset shapes and basic numerical sanity."""
    if theta.ndim != 2:
        raise ValueError(f"Expected theta to have shape (N, {cfg.theta_dim}), got {theta.shape}")
    if x.ndim != 3:
        raise ValueError(
            f"Expected x to have shape (N, {cfg.num_time_points}, {cfg.num_features}), got {x.shape}"
        )
    if theta.shape[0] != x.shape[0]:
        raise ValueError(
            f"theta and x must have the same number of examples, got {theta.shape[0]} and {x.shape[0]}"
        )
    if theta.shape[1] != cfg.theta_dim:
        raise ValueError(f"Expected theta.shape[1] == {cfg.theta_dim}, got {theta.shape[1]}")
    if x.shape[1] != cfg.num_time_points or x.shape[2] != cfg.num_features:
        raise ValueError(
            "Expected x shape (N, "
            f"{cfg.num_time_points}, {cfg.num_features}), got {x.shape}"
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains NaN or inf values.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or inf values.")

def load_dataset(dataset_path: str | Path, cfg: NPEConfig) -> Tuple[Tensor, Tensor]:
    """Load a precomputed dataset from an .npz file.

    Required arrays in the .npz:
        theta: shape (N, 15)
        x:     shape (N, 32, 13)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. "
            "Create an .npz file with arrays named 'theta' and 'x'."
        )

    data = np.load(dataset_path)
    if "theta" not in data or "x" not in data:
        raise KeyError(
            f"Dataset file {dataset_path} must contain arrays named 'theta' and 'x'."
        )

    theta = np.asarray(data["theta"], dtype=np.float32)
    x = np.asarray(data["x"], dtype=np.float32)
    validate_dataset(theta, x, cfg)

    theta_tensor = torch.from_numpy(theta)
    x_tensor = torch.from_numpy(x)
    return theta_tensor, x_tensor


def normalize_observations(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Standardize x across the training set featurewise."""
    mean = x.mean(dim=(0, 1), keepdim=True)
    std = x.std(dim=(0, 1), keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    x_normalized = (x - mean) / std
    return x_normalized, mean, std


def run_npe_training(theta: Tensor, x: Tensor, cfg: NPEConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    print(f"\nTraining on device: {device.upper()}")

    if device == "cuda":
        cfg.training_batch_size = 4096
        print(f"GPU Detected: Cranking batch size to {cfg.training_batch_size}")

    prior = make_prior(cfg, device=device)

    theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)

    print("Stacking clean trajectories with Gaussian noise...")
    noise_std = 0.005 * torch.std(x, dim=0)
    x_noisy = x + torch.randn_like(x) * noise_std
    
    theta_expanded = torch.cat([theta, theta], dim=0)
    x_expanded = torch.cat([x, x_noisy], dim=0)
    
    x_normalized, x_mean, x_std = normalize_observations(x_expanded)

    embedding_net = TrajectoryEmbeddingNet(
        num_features=cfg.num_features,
        embedding_dim=cfg.sequence_embedding_dim,
    )

    density_estimator_build_fun = posterior_nn(
        model="nsf",
        hidden_features=cfg.hidden_features,
        num_transforms=cfg.num_transforms,
        embedding_net=embedding_net,
        dropout_probability=0.05,
        use_batch_norm=True,     
    )

    inference = NPE(
        prior=prior, 
        density_estimator=density_estimator_build_fun,
        device=device 
    )
    
    print(f"Starting training loop with {len(theta_expanded)} total augmented samples...")
    density_estimator = inference.append_simulations(theta_expanded, x_normalized).train(
        training_batch_size=cfg.training_batch_size,
        learning_rate=cfg.learning_rate,
        validation_fraction=cfg.validation_fraction,
        show_train_summary=True,
        stop_after_epochs=20,   
        max_num_epochs=1000,     
    )
    
    posterior = inference.build_posterior(density_estimator)
    return posterior, prior, x_mean, x_std

def condition_posterior_on_example(
    posterior,
    x_example: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Sample from q(theta | x_example) and handle GPU/CPU transfers."""
    if x_example.ndim != 2:
        raise ValueError(f"Expected x_example shape (32, 13), got {tuple(x_example.shape)}")
        
    device = getattr(posterior, "_device", "cpu")
    if device is None:
        device = "cpu"
    
    x_example_device = x_example.to(device)
    x_mean_device = x_mean.to(device)
    x_std_device = x_std.to(device)
    
    x_normalized = (x_example_device.unsqueeze(0) - x_mean_device) / x_std_device
    x_normalized = x_normalized.squeeze(0)
    
    samples = posterior.sample((num_samples,), x=x_normalized, show_progress_bars=False)
    
    return samples.cpu()


def posterior_summary(samples: Tensor, true_theta: Tensor | None = None) -> None:
    """Print posterior mean and central 90% intervals."""
    mean = samples.mean(dim=0)
    q05 = torch.quantile(samples, 0.05, dim=0)
    q95 = torch.quantile(samples, 0.95, dim=0)

    print("\nPosterior summary:")
    for i, name in enumerate(PARAMETER_NAMES):
        line = (
            f"{name:>3}: mean={mean[i]: .4f}, "
            f"90% CI=[{q05[i]: .4f}, {q95[i]: .4f}]"
        )
        if true_theta is not None:
            line += f", true={true_theta[i]: .4f}"
        print(line)


def plot_posterior_marginals(
    samples: Tensor,
    true_theta: Tensor | None = None,
) -> None:
    """Plot 1D posterior marginals for all inferred parameters."""
    arr = samples.detach().cpu().numpy()

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i < len(PARAMETER_NAMES):
            ax.hist(arr[:, i], bins=40, density=True, alpha=0.7)
            if true_theta is not None:
                ax.axvline(float(true_theta[i]), linestyle="--")
            ax.set_title(PARAMETER_NAMES[i])
        else:
            ax.axis("off")
    fig.suptitle("Posterior marginals for inferred initial conditions and masses")
    fig.tight_layout()
    fig.savefig("./plots/posterior_marginals.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    set_seed(7)
    cfg = NPEConfig()

    print(f"Loading dataset from {cfg.dataset_path}...")
    theta, x = load_dataset(cfg.dataset_path, cfg)
    print(f"Loaded theta with shape {tuple(theta.shape)}")
    print(f"Loaded x with shape {tuple(x.shape)}")

    print("\nTraining NPE posterior estimator on precomputed simulations...")
    posterior, prior, x_mean, x_std = run_npe_training(theta, x, cfg)

    x_obs = x[0]
    true_theta = theta[0]
    posterior_samples = condition_posterior_on_example(
        posterior=posterior,
        x_example=x_obs,
        x_mean=x_mean,
        x_std=x_std,
        num_samples=cfg.num_posterior_samples,
    )

    print("\nConditioned on the first observation in the dataset.")
    print(f"Observation matrix shape = {tuple(x_obs.shape)}")
    posterior_summary(posterior_samples, true_theta=true_theta)

    plot_posterior_marginals(
        posterior_samples,
        true_theta=true_theta,
    )

if __name__ == "__main__":
    main()