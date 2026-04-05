from __future__ import annotations
import time
import glob
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import gc 

from npe import (
    NPEConfig,
    run_npe_training,
    set_seed,
)

def train_model(data_dir: str = "new_dataset", max_files: int = 5) -> None:
    """
    Load training shards efficiently from a single directory, stitch them together, 
    shuffle them thoroughly to prevent batch bias, and train the NPE model.
    """
    start_time = time.time()
    set_seed(42)

    regular_files = sorted([f for f in glob.glob(f"{data_dir}/train_batch_*.npz") if "chaotic" not in f])
    chaos_files = sorted(glob.glob(f"{data_dir}/train_batch_chaotic_*.npz"))
    
    print(f"Found {len(regular_files)} regular batches and {len(chaos_files)} chaotic batches.")

    all_files = regular_files + chaos_files

    if not all_files:
        raise FileNotFoundError(f"No training batches found in {data_dir}/.")
        
    # 12 hour limit timeguard    
    if max_files and len(all_files) > max_files:
        print(f"Limiting to first {max_files} batches to ensure training finishes safely.")
        all_files = all_files[:max_files]

    print(f"\nLoading {len(all_files)} training shards...")


    theta_tensors = []
    x_tensors = []

    for idx, f in enumerate(all_files, 1):
        print(f"  -> Loading file {idx}/{len(all_files)}: {os.path.basename(f)}...")
        data = np.load(f)
        
        theta_tensors.append(torch.from_numpy(data["theta"]).float())
        x_tensors.append(torch.from_numpy(data["x"]).float())
        
        del data
        gc.collect() 
        
    print("\nStitching tensors together...")
    theta_train = torch.cat(theta_tensors, dim=0)
    x_train = torch.cat(x_tensors, dim=0)
    
    del theta_tensors, x_tensors
    gc.collect()

    print("Shuffling the combined dataset...")
    num_samples = theta_train.shape[0]
    shuffle_indices = torch.randperm(num_samples)
    
    theta_train = theta_train[shuffle_indices]
    x_train = x_train[shuffle_indices]
    
    del shuffle_indices
    gc.collect()

    print(f"Final training theta shape: {theta_train.shape}")
    print(f"Final training x shape: {x_train.shape}")

    # Configure and Train
    print("\nInitializing PyTorch and SBI...")
    cfg = NPEConfig(dataset_path=data_dir)
    posterior, prior, x_mean, x_std = run_npe_training(theta_train, x_train, cfg)
    del prior

    # Save the trained model and normalization stats to disk
    sample_count_k = num_samples // 1000
    save_name = f"trained_npe_model_{sample_count_k}k.pt"
    
    torch.save({
        "posterior": posterior,
        "x_mean": x_mean,
        "x_std": x_std
    }, save_name)
    
    print(f"\nModel successfully saved to {save_name}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    train_model(data_dir="new_dataset", max_files=10)