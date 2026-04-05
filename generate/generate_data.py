import numpy as np
from scipy.integrate import solve_ivp

def three_body_ode(t, y, m1, m2, m3, G=1.0):
    """
    Computes the derivatives for the 2D planar three-body problem.
    State vector y: [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3]
    """
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = y
    
    # Distance components
    dx12, dy12 = x2 - x1, y2 - y1
    dx13, dy13 = x3 - x1, y3 - y1
    dx23, dy23 = x3 - x2, y3 - y2

    # Squared distances
    d12_sq = dx12**2 + dy12**2
    d13_sq = dx13**2 + dy13**2
    d23_sq = dx23**2 + dy23**2

    # Cubed distances (r^3)
    d12_cubed = d12_sq * np.sqrt(d12_sq)
    d13_cubed = d13_sq * np.sqrt(d13_sq)
    d23_cubed = d23_sq * np.sqrt(d23_sq)

    # Accelerations (using Newton's Law of Universal Gravitation)
    ax1 = G * (m2 * dx12 / d12_cubed + m3 * dx13 / d13_cubed)
    ay1 = G * (m2 * dy12 / d12_cubed + m3 * dy13 / d13_cubed)
    
    ax2 = G * (m1 * (-dx12) / d12_cubed + m3 * dx23 / d23_cubed)
    ay2 = G * (m1 * (-dy12) / d12_cubed + m3 * dy23 / d23_cubed)
    
    ax3 = G * (m1 * (-dx13) / d13_cubed + m2 * (-dx23) / d23_cubed)
    ay3 = G * (m1 * (-dy13) / d13_cubed + m2 * (-dy23) / d23_cubed)

    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

def collision_event(t, y, m1, m2, m3, G=1.0, threshold=1e-3):
    """
    Event function to stop the integrator if a collision is imminent.
    Returns 0 if any distance falls below the threshold.
    """
    x1, y1, x2, y2, x3, y3 = y[:6]
    
    d12 = (x2 - x1)**2 + (y2 - y1)**2
    d13 = (x3 - x1)**2 + (y3 - y1)**2
    d23 = (x3 - x2)**2 + (y3 - y2)**2
    
    min_dist_sq = min(d12, d13, d23)
    return min_dist_sq - threshold**2

# Tell solve_ivp to terminate when this event returns 0
collision_event.terminal = True 
collision_event.direction = -1

def generate_late_time_track(masses, init_positions, init_velocities, t_max=150.0, track_duration=5.0, num_points=32, G=1.0):
    """
    Simulates the system from t=0 to t_max, then extracts a random 32-point track 
    representing a late-time observation.
    """
    m1, m2, m3 = masses
    y0 = np.concatenate((init_positions.flatten(), init_velocities.flatten()))
    
    solution = solve_ivp(
        fun=three_body_ode,
        t_span=(0, t_max),
        y0=y0,
        method='DOP853',
        events=collision_event,
        dense_output=True, # allows continuous evaluation later
        args=(m1, m2, m3, G),
        rtol=1e-9,  
        atol=1e-12
    )
    
    # If it collided, we can only observe up to the collision time
    actual_t_max = solution.t[-1]
    
    # If the system collided before a single track_duration could complete, discard it
    if actual_t_max < track_duration:
        return None
        
    t_start = np.random.uniform(0, actual_t_max - track_duration)
    t_end = t_start + track_duration
    
    t_eval = np.linspace(t_start, t_end, num_points)
    
    y_obs = solution.sol(t_eval)
    
    y_transposed = y_obs.T
    t_col = t_eval.reshape(-1, 1)
    track_tensor = np.hstack((y_transposed, t_col))
    
    return track_tensor

def generate_n_samples(n=20):
    valid_samples = []
    attempts = 0
    
    print(f"Generating {n} valid late-time tracks...\n")
    
    while len(valid_samples) < n:
        attempts += 1
        
        # Generate random initial conditions
        masses = np.random.uniform(0.5, 2.5, size=3)
        init_positions = np.random.uniform(-5.0, 5.0, size=(3, 2))
        init_velocities = np.random.uniform(-2.0, 2.0, size=(3, 2))
        
        track = generate_late_time_track(
            masses=masses, 
            init_positions=init_positions, 
            init_velocities=init_velocities,
            t_max=150.0,
            track_duration=5.0,
            num_points=32,
            G=1.0
        )
        
        # Save track if no early collision
        if track is not None:
            valid_samples.append({
                'sample_id': len(valid_samples) + 1,
                'masses': masses,
                'initial_conditions': np.concatenate((init_positions.flatten(), init_velocities.flatten())),
                'track_shape': track.shape,
                'track_preview': track[:2, :] 
            })
            
    print(f"Successfully generated {n} samples in {attempts} attempts.\n")
    print("\n")
    return valid_samples

if __name__ == "__main__":
    samples = generate_n_samples(20)
    for sample in samples:
        print(f"Sample {sample['sample_id']}")
        print(f"Masses: {sample['masses'].round(3)}")
        print(f"Init Conditions (pos, vel): {sample['initial_conditions'].round(3)}")
        print(f"Output Tensor Shape: {sample['track_shape']}")
        print(f"Track Preview (First 2 of 32 points) [x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3, t]:\n{sample['track_preview'].round(3)}\n")