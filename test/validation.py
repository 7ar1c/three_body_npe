from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from sbi.analysis.plot import pp_plot_lc2st, sbc_rank_plot
from sbi.diagnostics import check_sbc
from sbi.diagnostics.lc2st import LC2ST

from generate.generate_data import three_body_ode


PARAMETER_NAMES = [
    "x1", "y1", "x2", "y2", "x3", "y3",
    "vx1", "vy1", "vx2", "vy2", "vx3", "vy3",
    "m1", "m2", "m3",
]

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class NormalizedPosteriorAdapter:
    """Wrap an sbi posterior so diagnostics can consume raw, unnormalized x."""

    def __init__(self, posterior, x_mean: torch.Tensor, x_std: torch.Tensor):
        self.posterior = posterior
        self._device = getattr(posterior, "_device", None) or "cpu"
        if self._device is None:
            self._device = "cpu"
        self.x_mean = x_mean.to(self._device)
        self.x_std = x_std.to(self._device)

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        x_device = x.to(self._device)
        return (x_device - self.x_mean) / self.x_std

    def sample(self, sample_shape, x: torch.Tensor, **kwargs):
        return self.posterior.sample(sample_shape, x=self._normalize_x(x), **kwargs)

    def sample_batched(self, sample_shape, x: torch.Tensor, **kwargs):
        return self.posterior.sample_batched(sample_shape, x=self._normalize_x(x), **kwargs)

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor, **kwargs):
        return self.posterior.log_prob(theta.to(self._device), x=self._normalize_x(x), **kwargs)


def _call_with_supported_kwargs(func, *args, **kwargs):
    signature = inspect.signature(func)
    supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **supported_kwargs)


def run_posterior_predictive_check(posterior, x_obs, x_mean, x_std, num_samples: int = 50, G: float = 1.0):
    print(f"\nRunning Posterior Predictive Check with {num_samples} physics simulations...")

    wrapped_posterior = NormalizedPosteriorAdapter(posterior, x_mean, x_std)
    x_obs_device = x_obs.to(wrapped_posterior._device)
    t_eval = x_obs_device[:, 12].detach().cpu().numpy()

    predicted_samples = wrapped_posterior.sample((num_samples,), x=x_obs_device, show_progress_bars=False)
    predicted_thetas = predicted_samples.detach().cpu().numpy()

    predicted_tracks = []
    for theta in predicted_thetas:
        pos, vel, masses = theta[:6], theta[6:12], theta[12:]
        y0 = np.concatenate((pos, vel))
        m1, m2, m3 = masses

        solution = solve_ivp(
            fun=three_body_ode,
            t_span=(0, t_eval[-1]),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            args=(m1, m2, m3, G),
            rtol=1e-9,
            atol=1e-12,
        )
        if solution.success and len(solution.t) == len(t_eval):
            predicted_tracks.append(solution.y.T)

    plt.figure(figsize=(10, 8))
    for track in predicted_tracks:
        plt.plot(track[:, 0], track[:, 1], color="red", alpha=0.1, linewidth=1)
        plt.plot(track[:, 2], track[:, 3], color="blue", alpha=0.1, linewidth=1)
        plt.plot(track[:, 4], track[:, 5], color="green", alpha=0.1, linewidth=1)

    x_obs_cpu = x_obs.cpu() if x_obs.device.type != 'cpu' else x_obs
    x_obs_np = x_obs_cpu.detach().numpy()
    plt.plot(x_obs_np[:, 0], x_obs_np[:, 1], color="darkred", linewidth=2, label="True Body 1")
    plt.plot(x_obs_np[:, 2], x_obs_np[:, 3], color="darkblue", linewidth=2, label="True Body 2")
    plt.plot(x_obs_np[:, 4], x_obs_np[:, 5], color="darkgreen", linewidth=2, label="True Body 3")
    plt.scatter(
        x_obs_np[0, [0, 2, 4]], x_obs_np[0, [1, 3, 5]],
        color="black", marker="x", zorder=5, label="Observation Start",
    )

    plt.title("Posterior Predictive Check: Reconstructed Orbits")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "posterior_predictive_check.png", dpi=300, bbox_inches="tight")
    plt.close() # Replaced plt.show()


def run_simulation_based_calibration(posterior, theta_test, x_test, x_mean, x_std, num_posterior_samples: int = 1000, max_sbc_samples: int = 250):
    num_test_samples = min(len(theta_test), max_sbc_samples)
    indices = torch.randperm(len(theta_test))[:num_test_samples]

    wrapped_posterior = NormalizedPosteriorAdapter(posterior, x_mean, x_std)
    diagnostic_device = wrapped_posterior._device

    thetas = theta_test[indices].detach().to(diagnostic_device)
    xs = x_test[indices].detach().to(diagnostic_device)

    print(f"\nRunning GPU-Accelerated Robust SBC on {num_test_samples} test samples...")

    all_ranks = []
    valid_thetas = []
    all_dap_samples = []

    for i in range(num_test_samples):
        try:

            x_input = xs[i].unsqueeze(0) if xs[i].dim() == 1 else xs[i]
            samples = wrapped_posterior.sample((num_posterior_samples,), x=x_input, show_progress_bars=False)
            
            if samples.dim() == 3: samples = samples.squeeze(0)
            
            truth_cpu = thetas[i].cpu()
            samples_cpu = samples.cpu()
            rank = (samples_cpu < truth_cpu).sum(dim=0)
            
            all_ranks.append(rank)
            valid_thetas.append(truth_cpu)
            all_dap_samples.append(samples_cpu[0])
            
        except Exception as e:
            if "discriminant" in str(e) or "NaN" in str(e): continue
            print(f"  [!] SBC skip at index {i}: {e}")

    ranks = torch.stack(all_ranks).float()
    thetas_cpu = torch.stack(valid_thetas)
    dap_samples = torch.stack(all_dap_samples) 
    
    print(f"SBC Complete. Successfully used {len(ranks)}/{num_test_samples} orbits.")


    check_stats = check_sbc(ranks, thetas_cpu, dap_samples, num_posterior_samples=num_posterior_samples)

    print("SBC diagnostics:")
    print(f"  KS p-values per parameter = {check_stats['ks_pvals'].numpy()}")

    for plot_type in ["hist", "cdf"]:
        fig, _ = sbc_rank_plot(ranks=ranks, num_posterior_samples=num_posterior_samples, plot_type=plot_type, num_bins=20)
        title = "SBI SBC Rank Histograms" if plot_type == "hist" else "SBI SBC Empirical CDFs"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"sbc_rank_{plot_type}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {"ranks": ranks, "check_stats": check_stats}


def run_expected_coverage_diagnostics(posterior, theta_test, x_test, x_mean, x_std, num_posterior_samples: int = 1000, max_coverage_samples: int = 250):
    from sbi.diagnostics import run_tarp, check_tarp
    from sbi.analysis.plot import plot_tarp
    
    num_test_samples = min(len(theta_test), max_coverage_samples)
    indices = torch.randperm(len(theta_test))[:num_test_samples]

    wrapped_posterior = NormalizedPosteriorAdapter(posterior, x_mean, x_std)
    diagnostic_device = wrapped_posterior._device

    thetas = theta_test[indices].detach().to(diagnostic_device)
    xs = x_test[indices].detach().to(diagnostic_device)

    print(f"\nRunning TARP on {num_test_samples} test samples...")

    valid_thetas = []
    valid_xs = []

    for i in range(num_test_samples):
        try:
            x_input = xs[i].unsqueeze(0) if xs[i].dim() == 1 else xs[i]
            _ = wrapped_posterior.sample((2,), x=x_input, show_progress_bars=False)
            
            valid_thetas.append(thetas[i]) 
            valid_xs.append(xs[i])
        except Exception as e:
            if "discriminant" in str(e) or "NaN" in str(e): continue

    valid_thetas_tensor = torch.stack(valid_thetas).to(diagnostic_device)
    valid_xs_tensor = torch.stack(valid_xs).to(diagnostic_device)
    
    print(f"Running SBI TARP on {len(valid_thetas_tensor)} safe orbits...")

    expected_coverage, nominal_coverage = run_tarp(
        valid_thetas_tensor, 
        valid_xs_tensor, 
        wrapped_posterior, 
        num_posterior_samples=num_posterior_samples,
        show_progress_bar=False
    )

    expected_coverage = expected_coverage.cpu()
    nominal_coverage = nominal_coverage.cpu()

    atc, ks_pval = check_tarp(expected_coverage, nominal_coverage)

    print("Expected coverage diagnostics:")
    print(f"  Area-to-curve (ATC) statistic = {float(atc):.6f}")
    print(f"  KS p-value                    = {float(ks_pval):.6f}")

    fig, ax = plot_tarp(expected_coverage, nominal_coverage)
    ax.set_title("SBI Expected Coverage (TARP)")
    fig.savefig(PLOTS_DIR / "expected_coverage_tarp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"expected_coverage": expected_coverage, "nominal_coverage": nominal_coverage}


def run_local_coverage_diagnostics(posterior, x_obs, theta_test, x_test, x_mean, x_std, num_calibration_samples: int = 256, num_eval_posterior_samples: int = 2000, conf_alpha: float = 0.05):
    if len(theta_test) < 2:
        raise ValueError("Need at least two held-out examples to run LC2ST.")

    max_available = max(1, len(theta_test) - 1)
    num_calibration_samples = min(num_calibration_samples, max_available)

    calibration_indices = torch.arange(1, 1 + num_calibration_samples)
    theta_cal = theta_test[calibration_indices].detach().cpu()
    x_cal_structured = x_test[calibration_indices].detach().cpu()
    x_cal = x_cal_structured.reshape(num_calibration_samples, -1)

    wrapped_posterior = NormalizedPosteriorAdapter(posterior, x_mean.cpu(), x_std.cpu())
    
    try:
        post_samples_cal = wrapped_posterior.sample_batched(
            (1,), x=x_cal_structured, max_sampling_batch_size=min(64, num_calibration_samples), show_progress_bars=False
        )[0].detach().cpu()
    except Exception:
        print("\n[!] Batched LC2ST sampling failed.")
        post_samples_cal_list = []
        for i in range(num_calibration_samples):
            try:
                samp = wrapped_posterior.sample((1,), x=x_cal_structured[i].unsqueeze(0), show_progress_bars=False).squeeze(0)
                post_samples_cal_list.append(samp.detach().cpu())
            except Exception:
                post_samples_cal_list.append(torch.zeros(1, theta_cal.shape[-1])) # Placeholder for failed cals
        post_samples_cal = torch.cat(post_samples_cal_list)

    print(f"\nRunning SBI local coverage diagnostics (L-C2ST) with {num_calibration_samples} calibration pairs...")

    lc2st = LC2ST(thetas=theta_cal, xs=x_cal, posterior_samples=post_samples_cal, classifier="mlp", num_ensemble=1)
    trained_clfs_null = lc2st.train_under_null_hypothesis() or getattr(lc2st, "trained_clfs", None)
    trained_clfs_data = lc2st.train_on_observed_data() or getattr(lc2st, "trained_clfs", None)

    x_obs_cpu_device = x_obs.cpu() if x_obs.device.type != 'cpu' else x_obs
    x_obs_structured = x_obs_cpu_device.detach()
    x_obs_cpu = x_obs_structured.reshape(1, -1)
    posterior_samples_obs = wrapped_posterior.sample(
        (num_eval_posterior_samples,), x=x_obs_structured.unsqueeze(0), show_progress_bars=False
    ).squeeze(0).detach().cpu()

    probs_data, _ = _call_with_supported_kwargs(lc2st.get_scores, theta_o=posterior_samples_obs, x_o=x_obs_cpu, trained_clfs=trained_clfs_data, return_probs=True)
    probs_null, _ = _call_with_supported_kwargs(lc2st.get_statistics_under_null_hypothesis, theta_o=posterior_samples_obs, x_o=x_obs_cpu, trained_clfs=trained_clfs_null, return_probs=True)
    t_data = _call_with_supported_kwargs(lc2st.get_statistic_on_observed_data, theta_o=posterior_samples_obs, x_o=x_obs_cpu, trained_clfs=trained_clfs_data)
    t_null = _call_with_supported_kwargs(lc2st.get_statistics_under_null_hypothesis, theta_o=posterior_samples_obs, x_o=x_obs_cpu, trained_clfs=trained_clfs_null)
    p_value = _call_with_supported_kwargs(lc2st.p_value, posterior_samples_obs, x_obs_cpu, trained_clfs_null=trained_clfs_null, trained_clfs_data=trained_clfs_data)
    reject = _call_with_supported_kwargs(lc2st.reject_test, posterior_samples_obs, x_obs_cpu, alpha=conf_alpha, trained_clfs_null=trained_clfs_null, trained_clfs_data=trained_clfs_data)

    print("L-C2ST diagnostics on the selected observation:")
    print(f"  Observed test statistic = {float(t_data):.6f}")
    print(f"  p-value                 = {float(p_value):.6f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t_null, bins=50, density=True, alpha=0.5, label="Null distribution")
    ax.axvline(t_data, color="red", linewidth=2, label="Observed statistic")
    q_low, q_high = np.quantile(t_null, [0.0, 1.0 - conf_alpha])
    ax.axvline(q_low, color="black", linestyle="--", alpha=0.7, label=f"{int((1.0 - conf_alpha) * 100)}% region")
    ax.axvline(q_high, color="black", linestyle="--", alpha=0.7)
    ax.set_xlabel("L-C2ST statistic")
    ax.set_ylabel("Density")
    ax.set_title("SBI Local Coverage Diagnostic (L-C2ST)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "local_coverage_lc2st_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig) 

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    pp_plot_lc2st(probs=[probs_data], probs_null=probs_null, conf_alpha=conf_alpha, labels=["Observed posterior samples"], colors=["red"], ax=ax)
    ax.set_title("L-C2ST P-P Plot")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "local_coverage_lc2st_pp_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"p_value": p_value, "reject": reject}