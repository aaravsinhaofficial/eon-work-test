import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from tqdm import trange
from pathlib import Path
from flygym import Fly, SingleFlySimulation, Camera
from flygym.arena import FlatTerrain
from flygym.examples.locomotion import PreprogrammedSteps
from flygym.examples.locomotion.cpg_controller import CPGNetwork

# Configuration
output_dir = Path("./foraging_simulation")
output_dir.mkdir(exist_ok=True, parents=True)
run_time = 10.0  # seconds
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

# Define correction parameters - REDUCED FOR STABILITY
correction_vectors = {
    "F": np.array([-0.02, 0, 0, -0.02, 0, 0.02, 0.02]),
    "M": np.array([-0.01, 0.001, 0.02, -0.015, 0, -0.015, 0.0]),
    "H": np.array([0, 0, 0, -0.015, 0, 0.008, -0.015]),
}
correction_rates = {
    "retraction": (400, 350),  # Reduced from (800, 700)
    "stumbling": (1100, 900),  # Reduced from (2200, 1800)
}
stumbling_force_threshold = -1
max_increment = 40 / timestep  # Reduced from 80
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

# Create custom arena with visible sugar patches
class VisibleSugarArena(FlatTerrain):
    def __init__(self, size=(20, 20)):
        super().__init__(size=size)
        self.size = size
        self.sugar_field = self._create_sugar_field()
        self._add_sugar_visuals()
        
    def _create_sugar_field(self):
        x = np.linspace(-self.size[0]/2, self.size[0]/2, 100)
        y = np.linspace(-self.size[1]/2, self.size[1]/2, 100)
        xx, yy = np.meshgrid(x, y)
        sugar_field = np.zeros_like(xx)
        num_patches = 5
        patch_centers = []
        for _ in range(num_patches):
            while True:
                cx = np.random.uniform(-self.size[0]/2.5, self.size[0]/2.5)
                cy = np.random.uniform(-self.size[1]/2.5, self.size[1]/2.5)
                too_close = any(
                    np.sqrt((cx - px)**2 + (cy - py)**2) < 5
                    for px, py in patch_centers
                )
                if not too_close:
                    patch_centers.append((cx, cy))
                    break
            sigma = np.random.uniform(1.5, 2.5)
            intensity = np.random.uniform(0.7, 1.0)
            patch = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            sugar_field = np.maximum(sugar_field, patch * intensity)
        if sugar_field.max() > 0:
            sugar_field = sugar_field / sugar_field.max()
        return sugar_field
    
    def _add_sugar_visuals(self):
        sugar_cmap = LinearSegmentedColormap.from_list(
            "sugar", ["#FFFF00", "#FFA500", "#FF0000"]
        )
        grid_size = 30
        x = np.linspace(-self.size[0]/2, self.size[0]/2, grid_size)
        y = np.linspace(-self.size[1]/2, self.size[1]/2, grid_size)
        for i in range(grid_size):
            for j in range(grid_size):
                conc = self.get_sugar_concentration(x[i], y[j])
                if conc < 0.05:
                    continue
                box_size = (self.size[0]/(grid_size*2), self.size[1]/(grid_size*2), 0.01)
                rgba = sugar_cmap(conc)
                self.root_element.worldbody.add(
                    'geom',
                    type='box',
                    size=box_size,
                    pos=(x[i], y[j], 0.01),
                    rgba=rgba,
                    name=f"sugar_{i}_{j}",
                    friction=(1, 0.005, 0.0001)
                )
    
    def get_sugar_concentration(self, x, y):
        xi = int(np.interp(x, [-self.size[0]/2, self.size[0]/2], [0, 99]))
        yi = int(np.interp(y, [-self.size[1]/2, self.size[1]/2], [0, 99]))
        xi = np.clip(xi, 0, 99)
        yi = np.clip(yi, 0, 99)
        return self.sugar_field[yi, xi]
    
    def visualize_sugar_field(self, fly_positions=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.sugar_field, extent=[-self.size[0]/2, self.size[0]/2,
                                            -self.size[1]/2, self.size[1]/2],
                   origin='lower', cmap='YlOrBr')
        plt.colorbar(label='Sugar Concentration')
        plt.title('Sugar Field with Fly Path')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        if fly_positions:
            path_x = [p[0] for p in fly_positions]
            path_y = [p[1] for p in fly_positions]
            plt.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.7)
            plt.plot(path_x[0], path_y[0], 'go', markersize=8, label='Start')
            plt.plot(path_x[-1], path_y[-1], 'ro', markersize=8, label='End')
            for i in range(0, len(path_x), 100):
                if i+10 < len(path_x):
                    dx = path_x[i+10] - path_x[i]
                    dy = path_y[i+10] - path_y[i]
                    plt.arrow(path_x[i], path_y[i], dx, dy,
                              shape='full', lw=0, length_includes_head=True,
                              head_width=0.5, head_length=1, alpha=0.5)
            plt.legend()
        plt.savefig(output_dir / "sugar_field_with_path.png")
        plt.close()

# Initialize simulation
fly = Fly(
    enable_adhesion=True,
    draw_adhesion=True,
    init_pose="stretch",
    control="position",
    contact_sensor_placements=contact_sensor_placements,
)

arena = VisibleSugarArena(size=(20, 20))

top_down_cam = Camera(
    camera_name="camera_top",
    attachment_point=fly.model.worldbody,
    play_speed=0.1
)

sim = SingleFlySimulation(
    fly=fly,
    cameras=[top_down_cam],
    timestep=timestep,
    arena=arena,
)

obs, info = sim.reset()
initial_position = obs["fly"][0].copy()

# Stumbling detection setup
detected_segments = ["Tibia", "Tarsus1", "Tarsus2"]
stumbling_sensors = {leg: [] for leg in preprogrammed_steps.legs}
for i, sensor_name in enumerate(fly.contact_sensor_placements):
    leg = sensor_name.split("/")[1][:2]
    segment = sensor_name.split("/")[1][2:]
    if segment in detected_segments:
        stumbling_sensors[leg].append(i)
stumbling_sensors = {k: np.array(v) for k, v in stumbling_sensors.items()}

# Correction states
retraction_correction = np.zeros(6)
stumbling_correction = np.zeros(6)
retraction_persistence_counter = np.zeros(6)

# Foraging parameters
sugar_threshold = 0.1
memory_size = 20
turn_strength = 0.8
speed_boost = 1.3

foraging_state = {
    "last_sugar_detected": -1000,
    "current_turn": 0,
    "slowing_down": False,
    "last_max_sugar": 0,
    "in_sugar_patch": False
}

fly_path = [initial_position.copy()]
sugar_concentrations = []
leg_sugar_readings = []

arena_boundaries = {
    "x_min": -arena.size[0]/2 + 1,
    "x_max": arena.size[0]/2 - 1,
    "y_min": -arena.size[1]/2 + 1,
    "y_max": arena.size[1]/2 - 1
}

camera_height = 20  # mm

# Main simulation loop
for k in trange(target_num_steps):
    fly_pos = obs["fly"][0]
    fly_path.append(fly_pos.copy())

    # Update camera
    camera_pos = np.array([fly_pos[0], fly_pos[1], fly_pos[2] + camera_height])
    sim.physics.model.cam_pos[0] = camera_pos
    sim.physics.model.cam_quat[0] = np.array([1, 0, 0, 0])

    # Boundary enforcement
    fly_pos[0] = np.clip(fly_pos[0], arena_boundaries["x_min"], arena_boundaries["x_max"])
    fly_pos[1] = np.clip(fly_pos[1], arena_boundaries["y_min"], arena_boundaries["y_max"])

    # Read leg sugar (Hz)
    leg_sugar = []
    for i, leg in enumerate(preprogrammed_steps.legs):
        leg_pos = obs["end_effectors"][i]
        sugar_conc = arena.get_sugar_concentration(leg_pos[0], leg_pos[1])
        leg_sugar.append(sugar_conc)
        if sugar_conc > 0.1:
            red_intensity = min(1.0, 0.5 + sugar_conc * 0.5)
            sim.fly.change_segment_color(sim.physics, f"{leg}Tarsus1", (red_intensity, 0, 0, 1))
        else:
            sim.fly.change_segment_color(sim.physics, f"{leg}Tarsus1", (0.5, 0.5, 0.5, 1))

    leg_sugar_readings.append(leg_sugar)
    max_sugar = np.max(leg_sugar)
    sugar_concentrations.append(max_sugar)

    # Normalize Hz → [0,1]
    norm_sugar = [min(1.0, s/300.0) for s in leg_sugar]
    max_sugar_norm = max(norm_sugar)

    # Memory
    sugar_memory = leg_sugar_readings[-memory_size:]

    # Foraging decisions
    if max_sugar_norm > sugar_threshold:
        foraging_state["last_sugar_detected"] = k
        foraging_state["slowing_down"] = True
        foraging_state["last_max_sugar"] = max_sugar_norm
        foraging_state["in_sugar_patch"] = True

        # P9 → slowdown factor
        p9_inhibition = max_sugar_norm
        speed_factor = 1.0 - 0.6 * p9_inhibition

        # DNa02 → steering
        max_leg_idx = np.argmax(norm_sugar)
        max_leg = preprogrammed_steps.legs[max_leg_idx]
        steer_mag = turn_strength * max_sugar_norm
        if max_leg.startswith("L"):
            foraging_state["current_turn"] = +steer_mag
        else:
            foraging_state["current_turn"] = -steer_mag

    elif k - foraging_state["last_sugar_detected"] < 400:
        foraging_state["slowing_down"] = True
        foraging_state["current_turn"] *= 0.97
        speed_factor = 1.0 - 0.6 * foraging_state["last_max_sugar"]
    else:
        foraging_state["slowing_down"] = False
        foraging_state["in_sugar_patch"] = False
        if np.random.rand() < 0.02:
            foraging_state["current_turn"] = turn_strength * 0.6 * (1 if np.random.rand() > 0.5 else -1)
        else:
            foraging_state["current_turn"] *= 0.95
        speed_factor = 1.0

    # Apply to CPG
    turn_bias = foraging_state["current_turn"]
    if foraging_state["slowing_down"]:
        speed_adjustment = speed_factor
    else:
        speed_adjustment = speed_boost if np.random.rand() < 0.08 else 1.0

    cpg_network.intrinsic_freqs = intrinsic_freqs * speed_adjustment

    # Retraction & stumbling detection
    end_effector_z = fly_pos[2] - obs["end_effectors"][:, 2]
    idxs = np.argsort(end_effector_z)
    if end_effector_z[idxs[-1]] > end_effector_z[idxs[-3]] + 0.05:
        leg_to_correct = idxs[-1]
        if retraction_correction[leg_to_correct] > persistence_init_thr * timestep:
            retraction_persistence_counter[leg_to_correct] = 1
    else:
        leg_to_correct = None

    retraction_persistence_counter[retraction_persistence_counter > 0] += 1
    retraction_persistence_counter[retraction_persistence_counter > retraction_persistence * timestep] = 0

    cpg_network.step()
    joints_angles = []
    adhesion_onoff = []
    all_net_corrections = []

    for i, leg in enumerate(preprogrammed_steps.legs):
        phase = cpg_network.curr_phases[i]
        if leg.startswith("L"):
            adjusted_phase = (phase + turn_bias) % (2 * np.pi)
        else:
            adjusted_phase = (phase - turn_bias) % (2 * np.pi)

        # Retraction correction
        if i == leg_to_correct or retraction_persistence_counter[i] > 0:
            increment = correction_rates["retraction"][0] * timestep
            retraction_correction[i] += increment
            sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (1, 0, 0, 1))
        else:
            decrement = correction_rates["retraction"][1] * timestep
            retraction_correction[i] = max(0, retraction_correction[i] - decrement)
            sim.fly.change_segment_color(sim.physics, f"{leg}Tibia", (0.5, 0.5, 0.5, 1))

        # Stumbling correction
        forces = obs["contact_forces"][stumbling_sensors[leg], :]
        ori = obs["fly_orientation"]
        proj = np.dot(forces, ori)
        if (proj < stumbling_force_threshold).any():
            inc = correction_rates["stumbling"][0] * timestep
            stumbling_correction[i] += inc
            if retraction_correction[i] <= 0:
                sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (1, 0, 0, 1))
        else:
            dec = correction_rates["stumbling"][1] * timestep
            stumbling_correction[i] = max(0, stumbling_correction[i] - dec)
            sim.fly.change_segment_color(sim.physics, f"{leg}Femur", (0.5, 0.5, 0.5, 1))

        # Net correction
        net = retraction_correction[i] if retraction_correction[i] > 0 else stumbling_correction[i]
        net = np.clip(net, 0, max_increment * timestep)
        if leg[0] == "R":
            net *= right_leg_inversion[i]
        net *= step_phase_gain[leg](adjusted_phase)
        all_net_corrections.append(net)

        angles = preprogrammed_steps.get_joint_angles(
            leg, adjusted_phase, cpg_network.curr_magnitudes[i]
        )
        angles = np.clip(angles + net * correction_vectors[leg[1]], -0.5, 0.5)
        joints_angles.append(angles)

        adhesion_onoff.append(preprogrammed_steps.get_adhesion_onoff(leg, adjusted_phase))

    action = {
        "joints": np.concatenate(joints_angles),
        "adhesion": np.array(adhesion_onoff).astype(int),
    }

    try:
        obs, reward, terminated, truncated, info = sim.step(action)
        sim.render()
    except Exception as e:
        print(f"Physics error at step {k}: {e}")
        break

    if not foraging_state["slowing_down"]:
        foraging_state["current_turn"] *= 0.95

# Save outputs
if sim.cameras:
    video_path = output_dir / "foraging_simulation_with_visible_sugar.mp4"
    top_down_cam.save_video(video_path)
arena.visualize_sugar_field(fly_path)

# Plot sugar concentration over time
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(sugar_concentrations)) * timestep, sugar_concentrations)
plt.xlabel("Time (s)")
plt.ylabel("Max Sugar Concentration per Leg")
plt.title("Sugar Detection During Foraging")
plt.savefig(output_dir / "sugar_concentration_over_time.png")
plt.close()

print("Simulation complete!")
if sim.cameras:
    print(f"Top-down video saved to: {video_path}")
print(f"Final position: {fly_path[-1]}")
print(f"Distance traveled: {np.linalg.norm(fly_path[-1][:2] - initial_position[:2]):.2f} mm")
