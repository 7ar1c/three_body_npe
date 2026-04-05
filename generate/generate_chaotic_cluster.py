from pathlib import Path
import multiprocessing

from generate.generate_close_encounter import generate_chaotic_dataset 
from npe import set_seed

def main():
    CORE_CAP = 60 
    available_cpus = multiprocessing.cpu_count()
    use_cores = min(CORE_CAP, available_cpus - 4)
    
    set_seed(42)
    
    samples_per_batch = 10000 # 10k chaotic test samples for evaluation
    total_training_batches = 5  
    total_test_batches = 1
    
    data_dir = Path("dataset_chaotic")
    data_dir.mkdir(exist_ok=True)

    print(f"Starting targeted chaotic generation using {use_cores} cores.")
    
    print(f"\nGenerating {total_training_batches} Chaotic Training Batches...")
    for i in range(1, total_training_batches + 1):
        save_path = data_dir / f"train_batch_chaotic_{i:02d}.npz"
        
        if not save_path.exists():
            print(f"-> Generating Train Batch {i}/{total_training_batches}...")
            generate_chaotic_dataset(
                n_samples=samples_per_batch,
                save_path=str(save_path),
                t_max=150.0,
                track_duration=5.0,
                num_points=32,
                G=1.0,
                encounter_threshold=0.5,
                n_cores=use_cores
            )
        else:
            print(f"Train Batch {i} already exists, skipping.")

    # Chaotic test batches
    print(f"\nGenerating {total_test_batches} Chaotic Test Batches...")
    for i in range(1, total_test_batches + 1):
        save_path = data_dir / f"test_batch_chaotic_{i:02d}.npz"
        
        if not save_path.exists():
            print(f"-> Generating Test Batch {i}/{total_test_batches}...")
            generate_chaotic_dataset(
                n_samples=samples_per_batch,
                save_path=str(save_path),
                t_max=150.0,
                track_duration=5.0,
                num_points=32,
                G=1.0,
                encounter_threshold=0.5,
                n_cores=use_cores
            )
        else:
            print(f"-> Test Batch {i} already exists. Skipping.")

if __name__ == "__main__":
    main()