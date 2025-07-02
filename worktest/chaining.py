from pathlib import Path
import numpy as np
from tqdm import trange
from flygym import Camera, Simulation
from flygym.examples.locomotion import HybridTurningFly

# Configuration
output_dir = Path("outputs/advanced_vision/")
output_dir.mkdir(parents=True, exist_ok=True)
timestep = 1e-4
run_time = 2.0  # Simulation time
total_steps = int(run_time / timestep)

# Contact sensor configuration
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

# Create three flies in a chain formation
leader_fly = HybridTurningFly(
    name="leader",
    enable_adhesion=True,
    contact_sensor_placements=contact_sensor_placements,
    seed=0,
    timestep=timestep,
    spawn_pos=(3, 3, 0.5),
    spawn_orientation=(0, 0, -np.pi / 2),  # Initial heading: downward
)

follower_fly = HybridTurningFly(
    name="follower",
    enable_adhesion=True,
    contact_sensor_placements=contact_sensor_placements,
    seed=0,
    timestep=timestep,
    spawn_pos=(3, 0, 0.5),
    spawn_orientation=(0, 0, -np.pi / 2),  # Initial heading: downward
)

tail_fly = HybridTurningFly(
    name="tail",
    enable_adhesion=True,
    contact_sensor_placements=contact_sensor_placements,
    seed=0,
    timestep=timestep,
    spawn_pos=(3, -3, 0.5),
    spawn_orientation=(0, 0, -np.pi / 2),  # Initial heading: downward
)

# Create top-down camera
top_down_cam = Camera(
    camera_name="camera_top",
    attachment_point=leader_fly.model.worldbody,
    play_speed=0.1
)

# Create simulation
sim = Simulation(
    flies=[leader_fly, follower_fly, tail_fly],
    cameras=[top_down_cam],
    timestep=timestep,
)

# Run simulation with following behavior
obs, info = sim.reset(seed=0)

# Following parameters
follow_distance = 1.5  # Desired distance between flies
follow_gain = 3.0      # Control gain for turning
min_speed = 0.2        # Minimum forward speed

# Initialize previous positions for heading calculation
prev_follower_pos = obs["follower"]["fly"][0][:2].copy()
prev_tail_pos = obs["tail"]["fly"][0][:2].copy()

for _ in trange(total_steps, desc="Simulating flies following each other"):
    # Get current positions (x, y, z)
    leader_pos = obs["leader"]["fly"][0]
    follower_pos = obs["follower"]["fly"][0]
    tail_pos = obs["tail"]["fly"][0]
    
    # ===== LEADER BEHAVIOR =====
    time_in_cycle = sim.curr_time % 4.0  # 4-second cycle
    if time_in_cycle < 1.0:
        leader_action = np.array([0.0, 1.0])  # Forward
    elif time_in_cycle < 2.0:
        leader_action = np.array([-1.0, 0.8])  # Left turn
    elif time_in_cycle < 3.0:
        leader_action = np.array([0.0, 1.0])  # Forward
    else:
        leader_action = np.array([1.0, 0.8])  # Right turn
    
    # ===== FOLLOWER BEHAVIOR =====
    # Calculate current heading from velocity
    follower_vel = follower_pos[:2] - prev_follower_pos
    follower_speed = np.linalg.norm(follower_vel)
    
    if follower_speed > 1e-5:
        follower_heading = np.arctan2(follower_vel[1], follower_vel[0])
    else:
        # Fallback to initial heading if not moving
        follower_heading = -np.pi / 2
    
    # Calculate direction to leader
    direction_to_leader = leader_pos[:2] - follower_pos[:2]
    dist_to_leader = np.linalg.norm(direction_to_leader)
    
    if dist_to_leader > 0:
        direction_to_leader /= dist_to_leader
        desired_heading = np.arctan2(direction_to_leader[1], direction_to_leader[0])
        
        # Calculate smallest angle difference (-π to π)
        angle_error = (desired_heading - follower_heading + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate actions
        turn = np.clip(angle_error * follow_gain, -1.0, 1.0)
        forward = np.clip((dist_to_leader - follow_distance) * 0.5, min_speed, 1.0)
    else:
        turn, forward = 0.0, min_speed
    
    follower_action = np.array([turn, forward])
    
    # ===== TAIL BEHAVIOR =====
    # Calculate current heading from velocity
    tail_vel = tail_pos[:2] - prev_tail_pos
    tail_speed = np.linalg.norm(tail_vel)
    
    if tail_speed > 1e-5:
        tail_heading = np.arctan2(tail_vel[1], tail_vel[0])
    else:
        tail_heading = -np.pi / 2  # Fallback
    
    # Calculate direction to follower
    direction_to_follower = follower_pos[:2] - tail_pos[:2]
    dist_to_follower = np.linalg.norm(direction_to_follower)
    
    if dist_to_follower > 0:
        direction_to_follower /= dist_to_follower
        desired_heading = np.arctan2(direction_to_follower[1], direction_to_follower[0])
        
        # Calculate smallest angle difference
        angle_error = (desired_heading - tail_heading + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate actions
        turn = np.clip(angle_error * follow_gain, -1.0, 1.0)
        forward = np.clip((dist_to_follower - follow_distance) * 0.5, min_speed, 1.0)
    else:
        turn, forward = 0.0, min_speed
    
    tail_action = np.array([turn, forward])
    
    # Update previous positions for next iteration
    prev_follower_pos = follower_pos[:2].copy()
    prev_tail_pos = tail_pos[:2].copy()
    
    # Step the simulation
    obs, _, _, _, info = sim.step({
        "leader": leader_action,
        "follower": follower_action,
        "tail": tail_action,
    })
    sim.render()

# Save video
video_path = output_dir / "three_flies_following_corrected.mp4"
top_down_cam.save_video(video_path)
print(f"Simulation complete! Video saved to: {video_path}")