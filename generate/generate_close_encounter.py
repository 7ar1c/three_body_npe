import os
import numpy as np
import concurrent.futures
from functools import partial
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from generate.generate_data import three_body_ode, collision_event
from scipy.integrate import solve_ivp
from main import sample_parameters, build_theta

def create_close_encounter_event(threshold=0.5):

    def close_encounter(t, y, m1, m2, m3, G):
        d12 = np.sqrt((y[0]-y[2])**2 + (y[1]-y[3])**2)
        d13 = np.sqrt((y[0]-y[4])**2 + (y[1]-y[5])**2)
        d23 = np.sqrt((y[2]-y[4])**2 + (y[3]-y[5])**2)
        return min(d12, d13, d23) - threshold
        
    close_encounter.terminal = False 
    close_encounter.direction = -1   
    return close_encounter

def generate_chaotic_track(masses, init_positions, init_velocities, t_max=150.0, track_duration=5.0, num_points=32, G=1.0, encounter_threshold=0.5):
    """
    Simulates the system, filters for chaotic close encounters, and guarantees 
    the observation window brackets the interaction.
    """
    m1, m2, m3 = masses
    y0 = np.concatenate((init_positions.flatten(), init_velocities.flatten()))
    
    close_event = create_close_encounter_event(encounter_threshold)
    
    solution = solve_ivp(
        fun=three_body_ode,
        t_span=(0, t_max),
        y0=y0,
        method='DOP853',
        events=[collision_event, close_event], 
        dense_output=True, 
        args=(m1, m2, m3, G),
        rtol=1e-9,  
        atol=1e-12
    )
    
    encounter_times = solution.t_events[1]
    
    if len(encounter_times) == 0:
        return None 
        
    actual_t_max = solution.t[-1]
    if actual_t_max < track_duration:
        return None
        
    # Timestamp of the first time bodies cross the chaotic threshold
    t_encounter = encounter_times[0] 
    
    # Ensure the encounter is caught within the track_duration
    min_start = max(0, t_encounter - track_duration)
    max_start = min(t_encounter, actual_t_max - track_duration)
    
    t_start = np.random.uniform(min_start, max_start)
    t_end = t_start + track_duration
    
    t_eval = np.linspace(t_start, t_end, num_points)
    
    y_obs = solution.sol(t_eval)
    y_transposed = y_obs.T
    t_col = t_eval.reshape(-1, 1)
    track_tensor = np.hstack((y_transposed, t_col))
    
    return track_tensor

def _chaotic_worker_task(t_max, track_duration, num_points, G, encounter_threshold):
    """Executes a single chaotic simulation attempt."""
    masses, init_positions, init_velocities = sample_parameters()
    
    x = generate_chaotic_track(
        masses=masses,
        init_positions=init_positions,
        init_velocities=init_velocities,
        t_max=t_max,
        track_duration=track_duration,
        num_points=num_points,
        G=G,
        encounter_threshold=encounter_threshold
    )
    
    if x is not None:
        x = np.asarray(x, dtype=np.float32)
        if x.shape == (num_points, 13):
            theta = build_theta(init_positions, init_velocities, masses)
            return theta, x
    return None


def generate_chaotic_dataset(
    n_samples: int,
    save_path: str,
    t_max: float = 150.0,
    track_duration: float = 5.0,
    num_points: int = 32,
    G: float = 1.0,
    encounter_threshold: float = 0.5,
    n_cores=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dataset of chaotic simulations using a heavily guarded worker pool."""
    theta_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    attempts = 0
    
    if n_cores is None:
        n_cores = os.cpu_count() - 1

    print(f"Generating {n_samples} chaotic samples for {save_path} using {n_cores} parallel cores...")

    worker_func = partial(_chaotic_worker_task, t_max, track_duration, num_points, G, encounter_threshold)

    with ProcessPool(max_workers=n_cores) as pool:
        while len(theta_list) < n_samples:
            
            needed = n_samples - len(theta_list)
            # Since the rejection rate for chaos is high, we queue up way more attempts than we need so the CPUs never sit idle.
            batch_size = min(25000, int(needed * 20)) 
            if batch_size == 0: batch_size = 1
            
            # Submit batch with a strict 5-second timeout per simulation
            futures = [pool.schedule(worker_func, timeout=5) for _ in range(batch_size)]
            
            for future in concurrent.futures.as_completed(futures):
                attempts += 1
                try:
                    result = future.result()
                    if result is not None:
                        if len(theta_list) < n_samples:
                            theta_list.append(result[0])
                            x_list.append(result[1])
                            
                            if len(theta_list) % 500 == 0 or len(theta_list) == n_samples:
                                hit_rate = (len(theta_list) / attempts) * 100
                                print(f"Progress: {len(theta_list)}/{n_samples} valid samples collected "
                                      f"(Attempts: {attempts} | Hit Rate: {hit_rate:.2f}%)")
                                
                except TimeoutError:
                    # Simulation likely got stuck in an infinite gravity well, so pass
                    pass
                except Exception as e:
                    # For any other physics crashes
                    pass
                    
            if len(theta_list) >= n_samples:
                for f in futures:
                    f.cancel()
                break

    theta_array = np.stack(theta_list, axis=0)
    x_array = np.stack(x_list, axis=0)

    np.savez_compressed(save_path, theta=theta_array, x=x_array)
    
    print(f"Final shapes:\ntheta: {theta_array.shape}, x: {x_array.shape}\n")
    print("\n")

    return theta_array, x_array