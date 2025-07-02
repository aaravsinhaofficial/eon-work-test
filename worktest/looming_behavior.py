import numpy as np
import matplotlib.pyplot as plt
from gymnasium.utils.env_checker import check_env
from flygym import Fly, Camera
from flygym.arena import FlatTerrain, BaseArena
from flygym.examples.locomotion import HybridTurningController
from pathlib import Path
from dm_control import mjcf


class LoomingArena(BaseArena):
    def __init__(
        self,
        size=(300, 300),
        obj_radius=1,
        init_pos=(0, 5, 1),
        approach_speed=5,
        target_fly_name=None,
    ):
        super().__init__()
        self.init_pos = np.array([*init_pos], dtype="float32")
        self.obj_radius = obj_radius
        self.ball_pos = self.init_pos.copy()
        self.speed = approach_speed
        self.curr_time = 0
        self.target_fly_name = target_fly_name

        self.root_element = mjcf.RootElement()

        mat = self.root_element.asset.add("material", name="loom_material", reflectance=0.1)
        self.root_element.worldbody.add(
            "body", name="loom_sphere_mocap", mocap=True, pos=self.ball_pos, gravcomp=1
        )
        body = self.root_element.find("body", "loom_sphere_mocap")
        body.add(
            "geom",
            name="loom_sphere",
            type="sphere",
            size=(obj_radius,),
            rgba=(1, 0, 0, 1),
            material=mat,
        )

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def step(self, dt, physics):
        fly_body = physics.model.name2id(self.target_fly_name, 'body')
        fly_pos = physics.data.xpos[fly_body, :3]
        direction = fly_pos - self.ball_pos
        direction /= np.linalg.norm(direction[:2])
        self.ball_pos[:2] += direction[:2] * self.speed * dt
        physics.bind(self.root_element.find("body", "loom_sphere_mocap")).mocap_pos = self.ball_pos
        self.curr_time += dt

    def reset(self, physics):
        self.curr_time = 0
        self.ball_pos = self.init_pos.copy()
        physics.bind(self.root_element.find("body", "loom_sphere_mocap")).mocap_pos = self.ball_pos

    def _get_max_floor_height(self):
        return 0.0


class LoomingController(HybridTurningController):
    def __init__(
        self,
        loom_thresh=1.5,
        freeze_prob=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loom_thresh = loom_thresh
        self.prev_area = None
        self.responded = False
        self.action = None
        self.freeze_prob = freeze_prob

    def _compute_loom_area(self, vision):
        binary = vision.max(axis=-1) < 1.0
        return np.sum(binary)

    def step(self, control_signal):
        raw_obs, _, _, _, _ = super().step(control_signal)
        vision = raw_obs["vision"]
        area = self._compute_loom_area(vision)

        if self.prev_area is not None and not self.responded:
            if area / (self.prev_area + 1e-6) > self.loom_thresh:
                self.responded = True
                self.action = 'freeze' if np.random.rand() < self.freeze_prob else 'flee'
        self.prev_area = area

        if self.responded:
            if self.action == 'freeze':
                return np.zeros_like(control_signal), 0, False, False, {}
            else:
                coords = np.argwhere(vision.max(axis=-1) < 1.0)
                com = coords.mean(axis=0)
                left_turn = 1.2 if com[1] < 360 else 0.8
                right_turn = 0.8 if com[1] < 360 else 1.2
                flee_signal = np.array([left_turn, right_turn])
                return flee_signal, 0, False, False, {}

        return super().step(control_signal)


if __name__ == "__main__":
    output_dir = Path("./outputs/looming")
    output_dir.mkdir(exist_ok=True, parents=True)

    terrain = FlatTerrain()
    arena = LoomingArena(target_fly_name="fly_root")

    contact_sensor_placements = [
        f"{leg}{seg}" for leg in ["LF","LM","LH","RF","RM","RH"] 
        for seg in ["Tibia","Tarsus1","Tarsus2","Tarsus3","Tarsus4","Tarsus5"]
    ]
    fly = Fly(
        spawn_pos=(0, 0, 0.2),
        spawn_orientation=(0, 0, 0),
        contact_sensor_placements=contact_sensor_placements,
        enable_vision=True,
        enable_olfaction=False,
    )
    cam_params = {
        "mode": "fixed",
        "pos": (0, -10, 5),
        "euler": (np.deg2rad(45), 0, 0),
        "fovy": 45
    }
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name="loom_cam",
        camera_parameters=cam_params,
        play_speed=0.2,
    )
    sim = LoomingController(
        fly=fly,
        cameras=[cam],
        arena=arena,
        intrinsic_freqs=np.ones(6)*9,
        loom_thresh=1.5,
        freeze_prob=0.5,
    )

    check_env(sim)
    obs, _ = sim.reset()
    for _ in range(200):
        signal = np.ones(2)
        obs, _, _, _, _ = sim.step(signal)
        sim.render()
