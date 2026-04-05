from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import concurrent.futures
from functools import partial

from test.validation import run_posterior_predictive_check, run_simulation_based_calibration

from generate.generate_data import generate_late_time_track
from npe import (
    NPEConfig,
    condition_posterior_on_example,
    load_dataset,
    plot_posterior_marginals,
    posterior_summary,
    run_npe_training,
    set_seed,
)

from pebble import ProcessPool
from concurrent.futures import TimeoutError


PARAMETER_NAMES = [
    "x1", "y1", "x2", "y2", "x3", "y3",
    "vx1", "vy1", "vx2", "vy2", "vx3", "vy3",
    "m1", "m2", "m3",
]


def sample_parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample masses, initial positions, and initial velocities.
    """
    masses = np.random.uniform(0.5, 2.5, size=3).astype(np.float32)
    init_positions = np.random.uniform(-5.0, 5.0, size=(3, 2)).astype(np.float32)
    init_velocities = np.random.uniform(-2.0, 2.0, size=(3, 2)).astype(np.float32)
    return masses, init_positions, init_velocities


def build_theta(
    init_positions: np.ndarray,
    init_velocities: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Build the 15D parameter vector expected by npe.py.

    Order:
        [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3, m1, m2, m3]
    """
    theta = np.concatenate(
        [
            init_positions.reshape(-1),
            init_velocities.reshape(-1),
            masses.reshape(-1),
        ]
    ).astype(np.float32)
    return theta



def _worker_task(t_max, track_duration, num_points, G):
    """Executes a single simulation attempt."""
    masses, init_positions, init_velocities = sample_parameters()
    
    x = generate_late_time_track(
        masses=masses,
        init_positions=init_positions,
        init_velocities=init_velocities,
        t_max=t_max,
        track_duration=track_duration,
        num_points=num_points,
        G=G,
    )
    
    if x is not None:
        x = np.asarray(x, dtype=np.float32)
        if x.shape == (num_points, 13):
            theta = build_theta(init_positions, init_velocities, masses)
            return theta, x
    return None


def generate_dataset(
    n_samples: int,
    save_path: str,
    t_max: float = 150.0,
    track_duration: float = 5.0,
    num_points: int = 32,
    G: float = 1.0,
    n_cores=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of valid simulations using a heavily guarded worker pool."""
    theta_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    attempts = 0
    if n_cores is None:
        n_cores = os.cpu_count() - 1
        

    print(f"Generating {n_samples} samples for {save_path} using {n_cores} parallel cores...")

    worker_func = partial(_worker_task, t_max, track_duration, num_points, G)

    with ProcessPool(max_workers=n_cores) as pool:
        while len(theta_list) < n_samples:
            
            needed = n_samples - len(theta_list)
            batch_size = min(5000, int(needed * 1.5))
            if batch_size == 0: batch_size = 1
            
            futures = [pool.schedule(worker_func, timeout=5) for _ in range(batch_size)]
            
            for future in concurrent.futures.as_completed(futures):
                attempts += 1
                try:
                    result = future.result()
                    if result is not None:
                        if len(theta_list) < n_samples:
                            theta_list.append(result[0])
                            x_list.append(result[1])
                            
                            if len(theta_list) % 1000 == 0 or len(theta_list) == n_samples:
                                print(f"Progress: {len(theta_list)}/{n_samples} valid samples collected (Total attempts: {attempts})")
                                
                except TimeoutError:
                    # The simulation got stuck in an infinite gravity well.
                    pass
                except Exception as e:
                    # for any other physics crashes
                    pass
                    
            if len(theta_list) >= n_samples:
                # Cancel pending futures in the queue
                for f in futures:
                    f.cancel()
                break

    theta_array = np.stack(theta_list, axis=0)
    x_array = np.stack(x_list, axis=0)

    np.savez(save_path, theta=theta_array, x=x_array)
    print(f"\nSuccessfully saved dataset to {save_path}")
    print(f"Final shapes -> theta: {theta_array.shape}, x: {x_array.shape}\n")
    print("-" * 50)

    return theta_array, x_array


def evaluate_on_test_set(
    posterior,
    theta_test: torch.Tensor,
    x_test: torch.Tensor,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    num_posterior_samples: int,
    max_examples: int | None = None,
) -> None:
    """Evaluate posterior quality on a held-out test set."""
    device = theta_test.device
    
    num_examples = theta_test.shape[0]
    if max_examples is not None:
        num_examples = min(num_examples, max_examples)

    posterior_means = []
    covered_90 = []
    interval_widths = []
    
    # Tracking for the safety net
    successful_indices = []
    failed_evals = 0

    print(f"\nEvaluating on {num_examples} held-out test examples...")

    for idx in range(num_examples):
        try:
            samples_cpu = condition_posterior_on_example(
                posterior=posterior,
                x_example=x_test[idx],
                x_mean=x_mean,
                x_std=x_std,
                num_samples=num_posterior_samples,
            )
            samples = samples_cpu.to(device)

            mean = samples.mean(dim=0)
            q05 = torch.quantile(samples, 0.05, dim=0)
            q95 = torch.quantile(samples, 0.95, dim=0)
            truth = theta_test[idx]

            posterior_means.append(mean)
            covered_90.append(((truth >= q05) & (truth <= q95)).float())
            interval_widths.append(q95 - q05)
            
            successful_indices.append(idx)

            if len(successful_indices) % 100 == 0 or idx + 1 == num_examples:
                print(f"Evaluated {len(successful_indices)}/{num_examples} examples (Failed: {failed_evals})")

        except Exception as e:
            # Catch singularity / NaN crashes and keep moving
            failed_evals += 1
            continue

    if not successful_indices:
        raise RuntimeError("Failure: No successful posterior evaluations on the test set.")

    posterior_means_tensor = torch.stack(posterior_means, dim=0)
    covered_90_tensor = torch.stack(covered_90, dim=0)
    interval_widths_tensor = torch.stack(interval_widths, dim=0)

    truth_tensor = theta_test[successful_indices]

    errors = posterior_means_tensor - truth_tensor
    rmse = torch.sqrt(torch.mean(errors**2, dim=0))
    mae = torch.mean(torch.abs(errors), dim=0)
    coverage_90 = torch.mean(covered_90_tensor, dim=0)
    avg_interval_width = torch.mean(interval_widths_tensor, dim=0)

    overall_rmse = torch.sqrt(torch.mean(errors**2)).cpu().item()
    overall_mae = torch.mean(torch.abs(errors)).cpu().item()
    overall_coverage_90 = torch.mean(covered_90_tensor).cpu().item()

    print("\n--- Final Robust Evaluation Report ---")
    print(f"Total Orbits Attempted: {num_examples}")
    print(f"Successful Evaluations: {len(successful_indices)}")
    print(f"Chaotic Rejections:     {failed_evals} ({(failed_evals/num_examples)*100:.1f}%)")
    print(f"Overall posterior-mean RMSE: {overall_rmse:.4f}")
    print(f"Overall posterior-mean MAE:  {overall_mae:.4f}")
    print(f"Overall 90% coverage:       {overall_coverage_90:.4f}")

    print("\nPer-parameter test metrics:")
    rmse_cpu = rmse.cpu()
    mae_cpu = mae.cpu()
    coverage_90_cpu = coverage_90.cpu()
    avg_interval_width_cpu = avg_interval_width.cpu()
    
    # Import PARAMETER_NAMES locally or ensure it's available in scope
    PARAMETER_NAMES = ["x1", "y1", "x2", "y2", "x3", "y3", "vx1", "vy1", "vx2", "vy2", "vx3", "vy3", "m1", "m2", "m3"]
    for i, name in enumerate(PARAMETER_NAMES):
        print(
            f"{name:>3}: RMSE={rmse_cpu[i].item():.4f}, "
            f"MAE={mae_cpu[i].item():.4f}, "
            f"90% coverage={coverage_90_cpu[i].item():.4f}, "
            f"avg 90% width={avg_interval_width_cpu[i].item():.4f}"
        )


def train_and_evaluate(
    train_dataset_path: str,
    test_dataset_path: str,
    max_test_examples: int | None = 50,
) -> None:
    """Load datasets, train NPE, evaluate on test set, and inspect one posterior."""
    cfg = NPEConfig(dataset_path=train_dataset_path)

    # Detect available device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device.upper()}")

    theta_train, x_train = load_dataset(train_dataset_path, cfg)
    theta_test, x_test = load_dataset(test_dataset_path, cfg)
    
    # Move test data to GPU immediately for evaluation
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)

    print(f"Loaded training theta with shape {tuple(theta_train.shape)}")
    print(f"Loaded training x with shape {tuple(x_train.shape)}")
    print(f"Loaded test theta with shape {tuple(theta_test.shape)} on {device.upper()}")
    print(f"Loaded test x with shape {tuple(x_test.shape)} on {device.upper()}")

    posterior, prior, x_mean, x_std = run_npe_training(theta_train, x_train, cfg)
    del prior

    # Save the trained model and stats to disk
    torch.save({
        "posterior": posterior,
        "x_mean": x_mean,
        "x_std": x_std
    }, "trained_npe_model.pt")
    print("\nModel saved to trained_npe_model.pt")

    evaluate_on_test_set(
        posterior=posterior,
        theta_test=theta_test,
        x_test=x_test,
        x_mean=x_mean,
        x_std=x_std,
        num_posterior_samples=cfg.num_posterior_samples,
        max_examples=max_test_examples,
    )

    x_obs = x_test[0]
    true_theta = theta_test[0]

    posterior_samples = condition_posterior_on_example(
        posterior=posterior,
        x_example=x_obs,
        x_mean=x_mean,
        x_std=x_std,
        num_samples=cfg.num_posterior_samples,
    )

    print("\nConditioned on the first test observation.")
    print(f"Observation matrix shape = {tuple(x_obs.shape)}")

    posterior_summary(posterior_samples, true_theta=true_theta.cpu())
    plot_posterior_marginals(posterior_samples, true_theta=true_theta.cpu())
    
    # PPC on single test observation
    run_posterior_predictive_check(posterior, x_obs.cpu(), x_mean.cpu(), x_std.cpu(), num_samples=50)

    # SBC across entire set
    run_simulation_based_calibration(posterior, theta_test.cpu(), x_test.cpu(), x_mean.cpu(), x_std.cpu(), num_posterior_samples=1000)



def main() -> None:
    set_seed(7)

    train_dataset_path = Path("three_body_train_dataset.npz")
    test_dataset_path = Path("three_body_test_dataset.npz")

    n_train_samples = 100
    n_test_samples = 20

    generate_dataset(
        n_samples=n_train_samples,
        save_path=str(train_dataset_path),
        t_max=150.0,
        track_duration=5.0,
        num_points=32,
        G=1.0,
    )

    generate_dataset(
        n_samples=n_test_samples,
        save_path=str(test_dataset_path),
        t_max=150.0,
        track_duration=5.0,
        num_points=32,
        G=1.0,
    )

    print("\nStarting NPE training and held-out evaluation...")
    train_and_evaluate(
        train_dataset_path=str(train_dataset_path),
        test_dataset_path=str(test_dataset_path),
        max_test_examples=50,
    )


if __name__ == "__main__":
    main()