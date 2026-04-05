from pathlib import Path
import multiprocessing

from main import generate_dataset
from npe import set_seed

def main():
    CORE_CAP = 60 
    available_cpus = multiprocessing.cpu_count()
    use_cores = min(CORE_CAP, available_cpus - 4)
    
    set_seed(42)
    
    samples_per_batch = 100000
    total_training_batches = 0
    total_test_batches = 5
    
    data_dir = Path("dataset_2M")
    data_dir.mkdir(exist_ok=True)

    print(f"Starting 2M sample generation using {use_cores} cores.")
    
    # Training Batches
    print(f"\nGenerating {total_training_batches} Training Batches...")
    for i in range(1, total_training_batches + 1):
        save_path = data_dir / f"train_batch_{i:02d}.npz"
        
        if not save_path.exists():
            print(f"-> Generating Train Batch {i}/{total_training_batches}...")
            generate_dataset(
                n_samples=samples_per_batch,
                save_path=str(save_path),
                t_max=150.0,
                track_duration=5.0,
                num_points=32,
                G=1.0,
                n_cores=use_cores
            )
        else:
            print(f"-> Train Batch {i} already exists. Skipping.")

    # Test Batches
    print(f"\nGenerating {total_test_batches} Test Batches...")
    for i in range(1, total_test_batches + 1):
        save_path = data_dir / f"test_batch_{i:02d}.npz"
        
        if not save_path.exists():
            print(f"-> Generating Test Batch {i}/{total_test_batches}...")
            generate_dataset(
                n_samples=samples_per_batch,
                save_path=str(save_path),
                t_max=150.0,
                track_duration=5.0,
                num_points=32,
                G=1.0,
                n_cores=use_cores
            )
        else:
            print(f"-> Test Batch {i} already exists. Skipping.")

if __name__ == "__main__":
    main()