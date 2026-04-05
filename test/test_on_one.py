import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi import analysis as analysis
import random
from test.validation import run_posterior_predictive_check
from npe import plot_posterior_marginals, condition_posterior_on_example, NPEConfig

# 1. Configuration (Update these paths if your file names differ)
MODEL_PATH = "trained_npe_model_500k.pt" 
TEST_DATA_PATH = "dataset_2M/test_batch_01.npz" # Swap with your specific test set file
NUM_SAMPLES = 10000

def evaluate():
    print("Loading trained model and data...")
    # 1. Load the checkpoint dictionary and map it to your Mac's CPU
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # 2. Extract the model and the exact normalization stats used during training
    posterior = checkpoint["posterior"]
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    
    data = np.load(TEST_DATA_PATH)
    
    # 3. Pick a random example
    num_examples = data["theta"].shape[0]
    random_idx = np.random.randint(0, num_examples)
    print(f"Evaluating random test example at index: {random_idx}")
    
    # 4. Load the observation data
    theta_true = torch.as_tensor(data["theta"][random_idx], dtype=torch.float32)
    x_obs = torch.as_tensor(data["x"][random_idx], dtype=torch.float32)

    try:
        posterior_samples = condition_posterior_on_example(
        posterior=posterior,
        x_example=x_obs,
        x_mean=x_mean,
        x_std=x_std,
        num_samples=3000,
    )
        # Call your custom function. 
        # Since it returns None, it should handle saving/showing internally!
        plot_posterior_marginals(samples=posterior_samples, true_theta=theta_true)
        print("Custom marginal plot executed.")
    except Exception as e:
        print(f"Could not run custom marginal plot: {e}")
    

    # ---------------------------------------------------------
    # PLOT 3: Posterior Predictive Check (PPC)
    # ---------------------------------------------------------
    print("Generating Posterior Predictive Check...")
    try:
        # Call your custom function
        fig_ppc = run_posterior_predictive_check(
            posterior=posterior, 
            x_obs=x_obs, 
            x_mean=x_mean, 
            x_std=x_std, 
            num_samples=50, 
            G=1.0
        )
        
        # Save the figure returned by your function
        if fig_ppc is not None:
            fig_ppc.savefig("eval_3_ppc.png", dpi=300, bbox_inches='tight')
            plt.close(fig_ppc)
            print("Saved eval_3_ppc.png successfully.")
        else:
            print("Note: run_posterior_predictive_check did not return a figure object.")
            
    except Exception as e:
        print(f"Could not run PPC: {e}")

    print("\nEvaluation complete! Open the generated .png files to view results.")

if __name__ == "__main__":
    evaluate()