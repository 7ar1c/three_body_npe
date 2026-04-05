from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from test.validation import (
    run_expected_coverage_diagnostics,
    run_local_coverage_diagnostics,
    run_posterior_predictive_check,
    run_simulation_based_calibration,
)
from main import evaluate_on_test_set

# Everything needed from your neural network file
from npe import (
    NPEConfig,
    condition_posterior_on_example,
    load_dataset,
    plot_posterior_marginals,
    posterior_summary,
)


PARAMETER_NAMES = [
    "x1", "y1", "x2", "y2", "x3", "y3",
    "vx1", "vy1", "vx2", "vy2", "vx3", "vy3",
    "m1", "m2", "m3",
]

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def find_valid_observation_for_diagnostics(
    posterior,
    theta_test,
    x_test,
    x_mean,
    x_std,
    num_samples: int,
    max_tries: int,
):
    """Return the first test example for which posterior sampling succeeds."""
    max_tries = min(max_tries, len(x_test))
    for idx in range(max_tries):
        x_obs = x_test[idx]
        true_theta = theta_test[idx]
        try:
            posterior_samples = condition_posterior_on_example(
                posterior=posterior,
                x_example=x_obs,
                x_mean=x_mean,
                x_std=x_std,
                num_samples=num_samples,
            )
            return idx, x_obs, true_theta, posterior_samples
        except AssertionError as exc:
            print(f"Skipping example {idx} during posterior conditioning due to sampler assertion: {exc}")
        except Exception as exc:
            print(f"Skipping example {idx} during posterior conditioning due to sampling failure: {exc}")

    raise RuntimeError(
        f"Could not find a valid observation for diagnostics in the first {max_tries} test examples."
    )


def save_open_figures(prefix: str) -> None:
    """Save all currently open matplotlib figures with a shared filename prefix."""
    figure_numbers = plt.get_fignums()
    if not figure_numbers:
        print(f"No open matplotlib figures found for prefix '{prefix}'.")
        return

    for i, fig_num in enumerate(figure_numbers):
        fig = plt.figure(fig_num)
        suffix = "" if len(figure_numbers) == 1 else f"_{i}"
        fig.savefig(PLOTS_DIR / f"{prefix}{suffix}.png", dpi=300, bbox_inches="tight")

def evaluate_pretrained_model(
    test_dataset_path: str,
    model_path: str = "trained_npe_model.pt",
    max_test_examples: int | None = 50,
) -> None:
    """Load a previously trained NPE model and run the full evaluation suite."""
    cfg = NPEConfig(dataset_path=test_dataset_path)

    device = 'cpu'

    print(f"Loading test dataset from {test_dataset_path}...")
    theta_test, x_test = load_dataset(test_dataset_path, cfg)

    print(f"\nLoading trained model from {model_path} to {device.upper()}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    posterior = checkpoint["posterior"]
    
    posterior._device = device 
    
    x_mean = checkpoint["x_mean"].to(device)
    x_std = checkpoint["x_std"].to(device)

    print("\nStarting Evaluation...")

    eval_num_samples = min(cfg.num_posterior_samples, 200)
    eval_max_examples = min(max_test_examples or 50, 1000)
    try:
        evaluate_on_test_set(
            posterior=posterior,
            theta_test=theta_test,
            x_test=x_test,
            x_mean=x_mean,
            x_std=x_std,
            num_posterior_samples=eval_num_samples,
            max_examples=eval_max_examples,
        )
    except AssertionError as exc:
        print(
            "Standard numerical evaluation hit a posterior sampling assertion. "
            f"Continuing with diagnostics-only mode. Details: {exc}"
        )

    obs_idx, x_obs, true_theta, posterior_samples = find_valid_observation_for_diagnostics(
        posterior=posterior,
        theta_test=theta_test,
        x_test=x_test,
        x_mean=x_mean,
        x_std=x_std,
        num_samples=eval_num_samples,
        max_tries=eval_max_examples,
    )

    print(f"\nConditioned on test observation {obs_idx}.")
    print(f"Observation matrix shape = {tuple(x_obs.shape)}")
    posterior_summary(posterior_samples, true_theta=true_theta)
    marginal_plot_result = plot_posterior_marginals(posterior_samples, true_theta=true_theta)
    if hasattr(marginal_plot_result, "savefig"):
        marginal_plot_result.savefig(
            PLOTS_DIR / f"posterior_marginals_obs_{obs_idx}.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        save_open_figures(f"posterior_marginals_obs_{obs_idx}")

    # Run posterior predictive check on the same observation
    run_posterior_predictive_check(posterior, x_obs, x_mean, x_std, num_samples=50)

    # run sbc on the full test set (or a subset if max_test_examples is set)
    run_simulation_based_calibration(
        posterior,
        theta_test,
        x_test,
        x_mean,
        x_std,
        num_posterior_samples=1000,
        max_sbc_samples=max_test_examples or 250,
    )

    # Run global expected coverage diagnostics
    run_expected_coverage_diagnostics(
        posterior,
        theta_test,
        x_test,
        x_mean,
        x_std,
        num_posterior_samples=1000,
        max_coverage_samples=max_test_examples or 250,
    )

    # Run local coverage diagnostics on the same observation used for the PPC
    run_local_coverage_diagnostics(
        posterior,
        x_obs,
        theta_test,
        x_test,
        x_mean,
        x_std,
        num_calibration_samples=min(1000, max(1, len(theta_test) - 1)),
        num_eval_posterior_samples=1000,
        conf_alpha=0.05,
    )


if __name__ == "__main__":
    evaluate_pretrained_model(
        test_dataset_path="./dataset_2M/test_batch_02.npz",
        model_path="./npe_posterior_finetuned.pt",
        max_test_examples=100,
    )
