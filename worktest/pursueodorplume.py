import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path
from tqdm import trange
import cv2
from enum import Enum
from flygym import Fly, Camera
from flygym.examples.olfaction.plume_tracking_arena import OdorPlumeArena
from flygym.examples.olfaction import PlumeNavigationTask
from flygym.util import get_data_path
from dm_control.mujoco import Camera as DmCamera
from numba import njit, prange

# ==============================
# 1. Simulate Complex Odor Plume
# ==============================
@njit(parallel=True)
def _resample_plume_image(grid_idx_all, plume_grid):
    plume_img = np.zeros(grid_idx_all.shape[:2])
    for i in prange(grid_idx_all.shape[0]):
        for j in prange(grid_idx_all.shape[1]):
            x_idx = grid_idx_all[i, j, 0]
            y_idx = grid_idx_all[i, j, 1]
            if x_idx != -1:
                plume_img[i, j] = plume_grid[y_idx, x_idx]
    return plume_img

def converging_brownian_step(value_curr, center, gaussian_scale=1.0, convergence=0.5):
    gaussian_center = (center - value_curr) * convergence
    value_diff = np.random.normal(loc=gaussian_center, scale=gaussian_scale, size=value_curr.shape)
    return value_curr + value_diff

def step(velocity_prev, smoke_prev, noise, noise_magnitude=(0.1, 2), dt=1.0, inflow=None):
    smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt=dt) + inflow
    external_force = smoke_next * noise * np.array(noise_magnitude) @ velocity_prev
    velocity_tentative = flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt=dt) + external_force
    velocity_next, pressure = flow.fluid.make_incompressible(velocity_tentative)
    return velocity_next, smoke_next

# Parameters
simulation_time = 20.0
dt = 0.05
arena_size = (80, 60)
inflow_pos = (4, 30)
inflow_radius = 1
inflow_scaler = 0.2
velocity_grid_size = 0.5
smoke_grid_size = 0.25
simulation_steps = int(simulation_time / dt)
output_dir = Path("outputs/plume_tracking")
output_dir.mkdir(parents=True, exist_ok=True)

# Simulate Brownian wind
curr_wind = np.zeros(2)
wind_hist = [curr_wind.copy()]
for i in range(simulation_steps):
    curr_wind = converging_brownian_step(curr_wind, (0, 0), (1.2, 1.2), 1.0)
    wind_hist.append(curr_wind.copy())

# Initialize grids
import phi.flow as flow
velocity = flow.StaggeredGrid(
    values=(10.0, 0.0),
    extrapolation=flow.extrapolation.BOUNDARY,
    x=int(arena_size[0] / velocity_grid_size),
    y=int(arena_size[1] / velocity_grid_size),
    bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
)

smoke = flow.CenteredGrid(
    values=0.0,
    extrapolation=flow.extrapolation.BOUNDARY,
    x=int(arena_size[0] / smoke_grid_size),
    y=int(arena_size[1] / smoke_grid_size),
    bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
)

inflow = inflow_scaler * flow.field.resample(
    flow.Sphere(x=inflow_pos[0], y=inflow_pos[1], radius=inflow_radius),
    to=smoke,
    soft=True,
)

# Run simulation
smoke_hist = []
for i in trange(simulation_steps):
    velocity, smoke = step(
        velocity,
        smoke,
        wind_hist[i],
        dt=dt,
        inflow=inflow,
        noise_magnitude=(0.5, 100.0),
    )
    smoke_hist.append(smoke.values.numpy("y,x"))

# Interpolate plume
sim_timepoints = np.arange(0, simulation_time, step=dt)
smoke_hist_interp_fun = interp1d(sim_timepoints, smoke_hist, axis=0)
new_timepoints = np.linspace(0, simulation_time - dt, num=10000)
smoke_hist_interp = smoke_hist_interp_fun(new_timepoints)

# Save plume
with h5py.File(output_dir / "plume.hdf5", "w") as f:
    f["plume"] = np.stack(smoke_hist_interp).astype(np.float16)
    f["inflow_pos"] = inflow_pos
    f["inflow_radius"] = [inflow_radius]
    f["inflow_scaler"] = [inflow_scaler]

# ==============================
# 2. Navigation Controller Logic
# ==============================
class WalkingState(Enum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3

def get_vector_angle(v):
    return np.arctan2(v[1], v[0])

def to_probability(x):
    x += np.abs(np.min(x)) + 1
    return x / np.sum(x)

class SimplePlumeNavigationController:
    def __init__(self, timestep, wind_dir=[-1.0, 0.0], seed=0):
        self.timestep = timestep
        self.wind_dir = wind_dir
        np.random.seed(seed)
        
        self.dn_drives = {
            WalkingState.FORWARD: np.array([1.0, 1.0]),
            WalkingState.TURN_LEFT: np.array((-0.4, 1.2)),
            WalkingState.TURN_RIGHT: np.array((1.2, -0.4)),
            WalkingState.STOP: np.array((0.0, 0.0)),
        }
        
        self.accumulated_evidence = 0.0
        self.accumulation_decay = 0.0001
        self.accumulation_odor_gain = 0.05
        self.accumulation_threshold = 20.0
        self.default_decision_interval = 0.75
        self.since_last_decision_time = 0.0
        self.min_evidence = -1 * self.accumulation_decay * self.default_decision_interval / timestep
        self.dn_drive_update_interval = 0.1
        self.dn_drive_update_steps = int(self.dn_drive_update_interval / self.timestep)
        self.dn_drive = self.dn_drives[WalkingState.STOP]
        
        self.curr_state = WalkingState.STOP
        self.target_angle = np.nan
        self.to_upwind_angle = np.nan
        self.upwind_success = [0, 0]
        self.boundary_refractory_period = 1.0
        self.boundary_time = 0.0

    def get_target_angle(self):
        up_wind_angle = get_vector_angle(self.wind_dir) - np.pi
        to_upwind_angle = np.tanh(self.accumulated_evidence) * np.pi / 4 - np.pi / 4
        crosswind_success_proba = to_probability(self.upwind_success)
        to_upwind_angle = np.random.choice([-1, 1], p=crosswind_success_proba) * np.abs(to_upwind_angle)
        
        target_angle = up_wind_angle + to_upwind_angle
        if target_angle > np.pi:
            target_angle -= 2 * np.pi
        elif target_angle < -np.pi:
            target_angle += 2 * np.pi
        return target_angle, to_upwind_angle

    def angle_to_dn_drive(self, fly_orientation):
        fly_angle = get_vector_angle(fly_orientation)
        angle_diff = self.target_angle - fly_angle
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        if np.isnan(self.target_angle):
            return self.dn_drives[WalkingState.STOP], WalkingState.STOP
        elif angle_diff > np.deg2rad(10):
            return self.dn_drives[WalkingState.TURN_LEFT], WalkingState.TURN_LEFT
        elif angle_diff < -np.deg2rad(10):
            return self.dn_drives[WalkingState.TURN_RIGHT], WalkingState.TURN_RIGHT
        else:
            return self.dn_drives[WalkingState.FORWARD], WalkingState.FORWARD

    def step(self, fly_orientation, odor_intensities, close_to_boundary, curr_time):
        if self.boundary_time > 0.0:
            self.boundary_time += self.timestep
        elif self.boundary_time > self.boundary_refractory_period:
            self.boundary_time = 0.0

        boundary_inv = close_to_boundary and self.boundary_time == 0.0

        if (self.accumulated_evidence > self.accumulation_threshold or 
            self.since_last_decision_time > self.default_decision_interval or 
            boundary_inv):
            if boundary_inv:
                if self.to_upwind_angle < np.deg2rad(-45):
                    self.upwind_success[0] -= 10
                elif self.to_upwind_angle > np.deg2rad(45):
                    self.upwind_success[1] -= 10
                self.boundary_time += self.timestep
            else:
                if self.to_upwind_angle < np.deg2rad(-45):
                    self.upwind_success[0] += 1 if self.accumulated_evidence > self.min_evidence else -1
                elif self.to_upwind_angle > np.deg2rad(45):
                    self.upwind_success[1] += 1 if self.accumulated_evidence > self.min_evidence else -1

            self.target_angle, self.to_upwind_angle = self.get_target_angle()
            self.accumulated_evidence = 0.0
            self.since_last_decision_time = 0.0
        else:
            self.accumulated_evidence += (odor_intensities.sum() * self.accumulation_odor_gain - 
                                          self.accumulation_decay)
        
        if (np.rint(curr_time / self.timestep) % self.dn_drive_update_steps == 0 or boundary_inv):
            self.dn_drive, self.curr_state = self.angle_to_dn_drive(fly_orientation)

        self.since_last_decision_time += self.timestep
        return self.dn_drive

    def reset(self, seed=0):
        np.random.seed(seed)
        self.accumulated_evidence = 0.0
        self.since_last_decision_time = 0.0
        self.upwind_success = [0, 0]
        self.boundary_time = 0.0
        self.target_angle = np.nan
        self.to_upwind_angle = np.nan
        self.curr_state = WalkingState.STOP
        self.dn_drive = self.dn_drives[self.curr_state]

# ==================================
# 3. Visualization Helper Functions
# ==================================
def get_debug_str(accumulated_evidence, curr_angle, target_angle, crosswind_success_proba):
    crosswind_success_proba_str = " ".join([f"{co:.2f}" for co in crosswind_success_proba])
    return [
        f"Accumulated evidence: {accumulated_evidence:.2f}",
        f"Fly orientation: {np.rad2deg(curr_angle):.2f}",
        f"Target angle: {np.rad2deg(target_angle):.2f}",
        f"Crosswind success proba: {crosswind_success_proba_str}",
    ]

def get_walking_icons():
    icons_dir = get_data_path("flygym", "data") / "etc/locomotion_icons"
    icons = {}
    for key in ["forward", "left", "right", "stop"]:
        icon_path = icons_dir / f"{key}.png"
        icons[key] = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
    return {
        WalkingState.FORWARD: icons["forward"],
        WalkingState.TURN_LEFT: icons["left"],
        WalkingState.TURN_RIGHT: icons["right"],
        WalkingState.STOP: icons["stop"],
    }

def get_inflow_circle(inflow_pos, inflow_radius, camera_matrix):
    circle_x, circle_y = [], []
    for angle in np.linspace(0, 2 * np.pi + 0.01, num=50):
        circle_x.append(inflow_pos[0] + inflow_radius * np.cos(angle))
        circle_y.append(inflow_pos[1] + inflow_radius * np.sin(angle))

    xyz_global = np.array([circle_x, circle_y, np.zeros_like(circle_x)])
    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    xs, ys, s = camera_matrix @ corners_homogeneous
    x = np.rint(xs / s).astype(int)
    y = np.rint(ys / s).astype(int)
    return x, y

def render_overlay(rendered_img, accumulated_evidence, fly_orientation, 
                  target_angle, crosswind_success_proba, icon, window_size, inflow_x, inflow_y):
    if rendered_img is not None:
        sub_strings = get_debug_str(
            accumulated_evidence,
            get_vector_angle(fly_orientation),
            target_angle,
            crosswind_success_proba,
        )
        for j, sub_string in enumerate(sub_strings):
            rendered_img = cv2.putText(
                rendered_img,
                sub_string,
                (5, window_size[1] - (len(sub_strings) - j + 1) * 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        rendered_img[
            window_size[1] - 100 - icon.shape[1] : window_size[1] - 100,
            0 : icon.shape[1],
            :,
        ] = icon

        rendered_img = cv2.polylines(
            rendered_img,
            [np.array([list(zip(inflow_x, inflow_y))])],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2,
        )
    return rendered_img

def is_close_to_boundary(pos, arena_size, margin=5.0):
    return (pos[0] < margin or pos[0] > arena_size[0] - margin or
            pos[1] < margin or pos[1] > arena_size[1] - margin)

# ===========================
# 4. Run Complete Simulation
# ===========================
if __name__ == "__main__":
    # Parameters
    timestep = 1e-4
    run_time = 10.0
    np.random.seed(0)
    
    # Load plume
    plume_data_path = output_dir / "plume.hdf5"
    
    # Create arena
    main_camera_name = "birdeye_camera"
    arena = OdorPlumeArena(
        plume_data_path,
        main_camera_name=main_camera_name,
        plume_simulation_fps=800,
        dimension_scale_factor=0.25
    )
    
    # Create fly
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        enable_olfaction=True,
        enable_vision=False,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(65.0, 45.0, 0.25),  # Starting position
        spawn_orientation=(0, 0, -np.pi),
    )
    
    # Create controller
    wind_dir = [1.0, 0.0]  # Wind direction (right to left)
    ctrl = SimplePlumeNavigationController(timestep, wind_dir=wind_dir)
    
    # Setup camera
    cam_params = {
        "mode": "fixed",
        "pos": (
            0.50 * arena.arena_size[0],
            0.15 * arena.arena_size[1],
            1.00 * arena.arena_size[1],
        ),
        "euler": (np.deg2rad(15), 0, 0),
        "fovy": 60
    }
    
    cam = Camera(
        attachment_point=arena.root_element.worldbody,
        camera_name=main_camera_name,
        timestamp_text=False,
        camera_parameters=cam_params
    )
    
    # Create simulation
    sim = PlumeNavigationTask(
        fly=fly,
        arena=arena,
        cameras=[cam],
    )
    
    # Precompute inflow visualization
    dm_cam = DmCamera(
        sim.physics,
        camera_id=cam.camera_id,
        width=cam.window_size[0],
        height=cam.window_size[1],
    )
    camera_matrix = dm_cam.matrix
    arena_inflow_pos = np.array(inflow_pos) / arena.dimension_scale_factor * smoke_grid_size
    target_inflow_radius = 5.0
    inflow_x, inflow_y = get_inflow_circle(
        arena_inflow_pos,
        target_inflow_radius,
        camera_matrix,
    )
    
    # Get walking icons
    walking_icons = get_walking_icons()
    
    # Run simulation
    obs, info = sim.reset(0)
    target_num_steps = int(run_time / timestep)
    
    for i in trange(target_num_steps):
        fly_orientation = obs["fly_orientation"][:2]
        fly_orientation /= np.linalg.norm(fly_orientation)
        close_to_boundary = is_close_to_boundary(obs["fly"][0][:2], arena.arena_size)
        
        dn_drive = ctrl.step(
            fly_orientation, 
            obs["odor_intensity"], 
            close_to_boundary, 
            sim.curr_time
        )
        
        obs, reward, terminated, truncated, info = sim.step(dn_drive)
        
        # Update visualization
        icon = walking_icons[ctrl.curr_state][:, :, :3]
        rendered_img = sim.render()[0]
        rendered_img = render_overlay(
            rendered_img,
            ctrl.accumulated_evidence,
            fly_orientation,
            ctrl.target_angle,
            to_probability(ctrl.upwind_success),
            icon,
            cam.window_size,
            inflow_x,
            inflow_y,
        )
        
        if rendered_img is not None:
            cam._frames[-1] = rendered_img
        
        # Check termination conditions
        if np.linalg.norm(obs["fly"][0][:2] - arena_inflow_pos) < target_inflow_radius:
            print("SUCCESS: Fly reached odor source!")
            break
        elif truncated:
            print("WARNING: Fly went out of bounds!")
            break
    
    # Save result video
    output_vid = output_dir / "plume_navigation_simulation.mp4"
    cam.save_video(output_vid)
    print(f"Simulation complete! Video saved to: {output_vid}")