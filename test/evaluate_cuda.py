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
    max_test_examples: int | None = 50000,
) -> None:
    """Load a previously trained NPE model and run the full evaluation suite."""
    cfg = NPEConfig(dataset_path=test_dataset_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Check: Running evaluation on '{device.upper()}'")

    print(f"Loading test dataset from {test_dataset_path}...")
    theta_test, x_test = load_dataset(test_dataset_path, cfg)
    
    theta_test = theta_test.to(device)
    x_test = x_test.to(device)

    print(f"\nLoading trained model from {model_path} to {device.upper()}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    posterior = checkpoint["posterior"]
    
    posterior._device = device

    x_mean = checkpoint["x_mean"].cpu()
    x_std = checkpoint["x_std"].cpu()

    print("\nStarting Evaluation...")

    eval_num_samples = 10000
    eval_max_examples = max_test_examples
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


    diagnostics_num_posterior_samples = min(10000, cfg.num_posterior_samples)
    max_examples_cap = max_test_examples if max_test_examples is not None else len(theta_test)
    diagnostics_max_sbc_samples = min(max_examples_cap, 5000)

    diagnostics_max_coverage_samples = min(max_examples_cap, 1000)
    diagnostics_num_eval_posterior_samples = 10000

    if hasattr(posterior, "set_sample_with"):
        try:
            maybe_new_posterior = posterior.set_sample_with("mcmc")
            if maybe_new_posterior is not None:
                posterior = maybe_new_posterior
            print("Configured posterior diagnostics sampling with MCMC.")
        except Exception as exc:
            print(f"Could not switch posterior sampling to MCMC; using default sampler. Details: {exc}")


    theta_test_cpu = theta_test.cpu()
    x_test_cpu = x_test.cpu()
    obs_idx, x_obs, true_theta, posterior_samples = find_valid_observation_for_diagnostics(
        posterior=posterior,
        theta_test=theta_test_cpu,
        x_test=x_test_cpu,
        x_mean=x_mean,
        x_std=x_std,
        num_samples=eval_num_samples,
        max_tries=eval_max_examples,
    )

    print(f"\nConditioned on test observation {obs_idx}.")
    print(f"Observation matrix shape = {tuple(x_obs.shape)}")
    posterior_summary(posterior_samples, true_theta=true_theta)
    

    marginal_plot_result = plot_posterior_marginals(posterior_samples.cpu(), true_theta=true_theta.cpu())
    if hasattr(marginal_plot_result, "savefig"):
        marginal_plot_result.savefig(
            PLOTS_DIR / f"posterior_marginals_obs_{obs_idx}.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        save_open_figures(f"posterior_marginals_obs_{obs_idx}")


    run_posterior_predictive_check(posterior, x_obs.cpu(), x_mean, x_std, num_samples=50)

    print("\nStarting massive GPU Simulation-Based Calibration (SBC)...")
    run_simulation_based_calibration(
        posterior,
        theta_test_cpu,
        x_test_cpu,
        x_mean,
        x_std,
        num_posterior_samples=diagnostics_num_posterior_samples,
        max_sbc_samples=diagnostics_max_sbc_samples,
    )

    try:
        print("\nStarting global TARP coverage diagnostics...")
        run_expected_coverage_diagnostics(
            posterior,
            theta_test_cpu,
            x_test_cpu,
            x_mean,
            x_std,
            num_posterior_samples=diagnostics_num_posterior_samples,
            max_coverage_samples=diagnostics_max_coverage_samples,
        )
    except AssertionError as exc:
        print(
            "TARP coverage diagnostics hit an assertion error during rejection sampling. "
        )
    except Exception as exc:
        print(
            f"TARP coverage diagnostics failed with error: {exc}. "
            "Skipping TARP and continuing with L-C2ST."
        )

    print("\nStarting L-C2ST diagnostics...")
    run_local_coverage_diagnostics(
        posterior,
        x_obs,
        theta_test_cpu,
        x_test_cpu,
        x_mean,
        x_std,
        num_calibration_samples=min(1000, max(1, len(theta_test_cpu) - 1)),
        num_eval_posterior_samples=diagnostics_num_eval_posterior_samples,
        conf_alpha=0.05,
    )


if __name__ == "__main__":
    evaluate_pretrained_model(
        test_dataset_path="./dataset_chaotic/test_batch_chaotic_01.npz", 
        model_path="./trained_npe_model_450k.pt",
        max_test_examples=50000,
    )