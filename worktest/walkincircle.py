import numpy as np
from tqdm import trange
from flygym import Fly, ZStabilizedCamera
from flygym.examples.locomotion import HybridTurningController

# Simulation parameters
run_time = 5  # seconds
timestep = 1e-4
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

# Create fly model with contact sensors
fly = Fly(
    enable_adhesion=True,
    draw_adhesion=True,
    contact_sensor_placements=contact_sensor_placements,
    spawn_pos=(0, 0, 0.2),
)

# Create stabilized camera
cam = ZStabilizedCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_top",
    targeted_fly_names=fly.name,
    play_speed=0.1
)

# Create turning controller environment
nmf = HybridTurningController(
    fly=fly,
    cameras=[cam],
    timestep=timestep,
)

# Run simulation with turning action
obs, info = nmf.reset(seed=0)
for i in trange(int(run_time / nmf.timestep)):
    # Apply asymmetric control signal to induce turning
    # [left_signal, right_signal] - higher value on right causes left turn
    action = np.array([0.2, 1.2])
    
    # Step the simulation
    obs, reward, terminated, truncated, info = nmf.step(action)
    
    # Render the scene
    nmf.render()

# Save the video
cam.save_video("turning_circle.mp4")