import numpy as np
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from flygym import Fly, ZStabilizedCamera, SingleFlySimulation
from flygym.arena import MixedTerrain
from flygym.examples.locomotion import PreprogrammedSteps
from flygym.examples.locomotion.cpg_controller import CPGNetwork

# Configuration
output_dir = Path("./hybrid_controller_output")
output_dir.mkdir(exist_ok=True, parents=True)
run_time = 1.0  # seconds
timestep = 1e-4
target_num_steps = int(run_time / timestep)

# Initialize preprogrammed steps
preprogrammed_steps = PreprogrammedSteps()

# Initialize CPG network
intrinsic_freqs = np.ones(6) * 12
intrinsic_amps = np.ones(6) * 1
phase_biases = np.pi * np.array([
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0]
])
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20
cpg_network = CPGNetwork(
    timestep=timestep,
    intrinsic_freqs=intrinsic_freqs,
    intrinsic_amps=intrinsic_amps,
    coupling_weights=coupling_weights,
    phase_biases=phase_biases,
    convergence_coefs=convergence_coefs
)

# Define correction parameters
correction_vectors = {
    "F": np.array([-0.03, 0, 0, -0.03, 0, 0.03, 0.03]),
    "M": np.array([-0.015, 0.001, 0.025, -0.02, 0, -0.02, 0.0]),
    "H": np.array([0, 0, 0, -0.02, 0, 0.01, -0.02]),
}
correction_rates = {
    "retraction": (800, 700),
    "stumbling": (2200, 1800),
}
stumbling_force_threshold = -1
max_increment = 80 / timestep
retraction_persistence = 20 / timestep
persistence_init_thr = 20 / timestep
right_leg_inversion = [1, -1, -1, 1, -1, 1, 1]

# Initialize phase-dependent gains
step_phase_gain = {}
for leg in preprogrammed_steps.legs:
    swing_start, swing_end = preprogrammed_steps.swing_period[leg]
    step_points = [
        swing_start,
        np.mean([swing_start, swing_end]),
        swing_end + np.pi / 4,
        np.mean([swing_end, 2 * np.pi]),
        2 * np.pi
    ]
    preprogrammed_steps.swing_period[leg] = (swing_start, swing_end + np.pi / 4)
    increment_vals = [0, 0.8, 0, -0.1, 0]
    step_phase_gain[leg] = interp1d(
        step_points, increment_vals, kind="linear", fill_value="extrapolate"
    )

# Setup simulation
contact_sensor_placements = [
    f"{leg}{segment}"
    for leg in preprogrammed_steps.legs
    for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
]

fly = Fly(
    enable_adhesion=True,
    draw_adhesion=True,
    init_pose="stretch",
    control="position",
    contact_sensor_placements=contact_sensor_placements,
)
cam = ZStabilizedCamera(
    attachment_point=fly.model.worldbody,
    camera_name="camera_left",
    targeted_fly_names=fly.name,
    play_speed=0.1
)
arena = MixedTerrain()
sim = SingleFlySimulation(
    fly=fly,
    cameras=[cam],
    timestep=timestep,
    arena=arena,
)

# FIX: Correctly unpack the reset return values
obs, info = sim.reset()

# Setup stumbling detection
detected_segments = ["Tibia", "Tarsus1", "Tarsus2"]
stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}
for i, sensor_name in enumerate(fly.contact_sensor_placements):
    leg = sensor_name.split("/")[1][:2]  # e.g., "LF" for "Animat/LFTarsus1"
    segment = sensor_name.split("/")[1][2:]
    if segment in detected_segments:
        stumbling_sensors[leg].append(i)
stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}

# Initialize correction states
retraction_correction = np.zeros(6)
stumbling_correction = np.zeros(6)
retraction_persistence_counter = np.zeros(6)

# Main simulation loop
for k in trange(target_num_steps):
    # Retraction rule detection
    # FIX: Access dictionary keys correctly
    fly_pos = obs["fly"][0]  # [x, y, z]
    end_effectors_pos = obs["end_effectors"]  # [6, 3] array
    end_effector_z_pos = fly_pos[2] - end_effectors_pos[:, 2]
    
    end_effector_z_pos_sorted_idx = np.argsort(end_effector_z_pos)
    end_effector_z_pos_sorted = end_effector_z_pos[end_effector_z_pos_sorted_idx]
    if end_effector_z_pos_sorted[-1] > end_effector_z_pos_sorted[-3] + 0.05:
        leg_to_correct_retraction = end_effector_z_pos_sorted_idx[-1]
        if retraction_correction[leg_to_correct_retraction] > persistence_init_thr * timestep:
            retraction_persistence_counter[leg_to_correct_retraction] = 1
    else:
        leg_to_correct_retraction = None

    # Update persistence counter
    retraction_persistence_counter[retraction_persistence_counter > 0] += 1
    retraction_persistence_counter[retraction_persistence_counter > retraction_persistence * timestep] = 0

    # Step CPG network
    cpg_network.step()
    joints_angles = []
    adhesion_onoff = []
    all_net_corrections = []

    for i, leg in enumerate(preprogrammed_steps.legs):
        # Update retraction correction
        if i == leg_to_correct_retraction or retraction_persistence_counter[i] > 0:
            increment = correction_rates["retraction"][0] * timestep
            retraction_correction[i] += increment
            sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (1, 0, 0, 1))
        else:
            decrement = correction_rates["retraction"][1] * timestep
            retraction_correction[i] = max(0, retraction_correction[i] - decrement)
            sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (0.5, 0.5, 0.5, 1))

        # Update stumbling correction
        contact_forces = obs["contact_forces"][stumbling_sensors[leg], :]
        fly_orientation = obs["fly_orientation"]
        force_proj = np.dot(contact_forces, fly_orientation)
        if (force_proj < stumbling_force_threshold).any():
            increment = correction_rates["stumbling"][0] * timestep
            stumbling_correction[i] += increment
            if retraction_correction[i] <= 0:
                sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (1, 0, 0, 1))
        else:
            decrement = correction_rates["stumbling"][1] * timestep
            stumbling_correction[i] = max(0, stumbling_correction[i] - decrement)
            sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (0.5, 0.5, 0.5, 1))

        # Apply correction
        if retraction_correction[i] > 0:
            net_correction = retraction_correction[i]
            stumbling_correction[i] = 0
        else:
            net_correction = stumbling_correction[i]
            
        net_correction = np.clip(net_correction, 0, max_increment * timestep)
        if leg[0] == "R":  # Right legs need inversion
            net_correction *= right_leg_inversion[i]
            
        # Apply phase-dependent gain
        phase = cpg_network.curr_phases[i] % (2 * np.pi)
        net_correction *= step_phase_gain[leg](phase)
        
        # Get target angles from CPG and apply correction
        my_joints_angles = preprogrammed_steps.get_joint_angles(
            leg, phase, cpg_network.curr_magnitudes[i]
        )
        my_joints_angles += net_correction * correction_vectors[leg[1]]
        joints_angles.append(my_joints_angles)
        all_net_corrections.append(net_correction)

        # Get adhesion signal
        adhesion_onoff.append(preprogrammed_steps.get_adhesion_onoff(leg, phase))

    # Execute action
    action = {
        "joints": np.concatenate(joints_angles),
        "adhesion": np.array(adhesion_onoff).astype(int),
    }
    # FIX: Correctly handle step return values
    obs, reward, terminated, truncated, info = sim.step(action)
    sim.render()

# Save video
video_path = output_dir / "hybrid_controller_mixed_terrain.mp4"
cam.save_video(video_path)
print(f"Simulation complete! Video saved to: {video_path}")