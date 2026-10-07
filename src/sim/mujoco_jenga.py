"""MuJoCo contact-physics benchmark with the upstream SO-101 and RGB cameras.

The benchmark predicts tower failure after a short disturbance observation.
Forces are applied to blocks directly, not by a learned arm controller.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROBOT = Path(__file__).resolve().parents[2] / "data/assets/so101/so101_new_calib.xml"


def scene_xml(seed=0, layers=12, robot_path=ROBOT):
    rng = np.random.default_rng(seed)
    root = ET.parse(robot_path).getroot()
    root.find("compiler").set("meshdir", str(robot_path.parent / "assets"))
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", iterations="80")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="table", type="plane", size="1 1 .02", rgba=".16 .19 .22 1", friction=".7 .005 .0001")
    ET.SubElement(world, "light", pos=".2 -.4 1.5", dir="0 0 -1", diffuse=".8 .8 .8")
    # Fixed calibrated-view convention, randomized light and wood material.
    ET.SubElement(world, "body", name="look_at", pos=".28 0 .10")
    ET.SubElement(world, "camera", name="side", mode="targetbody", target="look_at", pos=".60 -.38 .32", fovy="38")
    ET.SubElement(world, "camera", name="overview", mode="targetbody", target="look_at", pos=".65 -.65 .62", fovy="48")
    ET.SubElement(world, "camera", name="overhead", pos=".28 0 .8", quat="1 0 0 0", fovy="40")
    friction = rng.uniform(.35, .8)
    for layer in range(layers):
        for slot in range(3):
            x, y = .28, 0.
            offset = (slot - 1) * .0255
            if layer % 2:
                x += offset
            else:
                y += offset
            x += rng.uniform(-.0007, .0007)
            y += rng.uniform(-.0007, .0007)
            name = f"block_{layer}_{slot}"
            b = ET.SubElement(world, "body", name=name, pos=f"{x} {y} {.0075 + layer * .0152}", euler=f"0 0 {np.pi / 2 if layer % 2 else 0}")
            ET.SubElement(b, "freejoint", name=name + "_joint")
            wood = rng.uniform(.7, .9)
            ET.SubElement(b, "geom", name=name + "_geom", type="box", size=".0375 .0125 .0075", mass=str(rng.uniform(.010, .014)),
                          rgba=f"{wood} {wood*.72} {wood*.42} 1", friction=f"{friction} .005 .0001", condim="4", solref=".006 1")
    return ET.tostring(root, encoding="unicode")


class JengaPhysics:
    def __init__(self, seed=0, layers=12, render_size=224):
        self.layers = layers
        self.model = mujoco.MjModel.from_xml_string(scene_xml(seed, layers))
        self.data = mujoco.MjData(self.model)
        # Park the arm away from the tower during the perception benchmark.
        self.data.qpos[self.model.joint("shoulder_pan").qposadr[0]] = -1.2
        self.data.ctrl[self.model.actuator("shoulder_pan").id] = -1.2
        self.renderer = None
        self.render_size = render_size
        self.block_ids = np.array([self.model.body(f"block_{l}_{s}").id for l in range(layers) for s in range(3)])
        self.top_ids = self.block_ids[-3:]
        self.step(.4)
        self.initial = self.data.xpos[self.block_ids].copy()

    def step(self, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def frame(self, camera="side"):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.render_size, self.render_size)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def failed(self):
        # Unselected upper blocks losing >3 cm height or >4 cm lateral position.
        pos = self.data.xpos[self.block_ids]
        height_loss = self.initial[:, 2] - pos[:, 2]
        drift = np.linalg.norm(pos[:, :2] - self.initial[:, :2], axis=1)
        return bool(np.any(height_loss > .03) or np.any(drift > .04))

    def observe_disturbance(self, force, direction):
        frames = []
        for t in range(4):
            frames.append(self.frame())
            self.data.xfrc_applied[:] = 0
            if t < 2:
                for bid in self.top_ids:
                    self.data.xfrc_applied[bid, :2] = np.array([np.cos(direction), np.sin(direction)]) * force / 3
            self.step(.02)
        self.data.xfrc_applied[:] = 0
        return np.stack(frames)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def generate(output, episodes=120, seed=1000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    attempt = 0
    while len(records) < episodes:
        i = len(records)
        episode_seed = seed + attempt
        attempt += 1
        if attempt > episodes * 10:
            raise RuntimeError("Too many already-failed observations; revise disturbance distribution")
        rng = np.random.default_rng(episode_seed)
        env = JengaPhysics(episode_seed)
        try:
            force = float(rng.uniform(0, .75))
            clip = env.observe_disturbance(force, float(rng.uniform(-np.pi, np.pi)))
            failed_before = env.failed()
            if failed_before:
                continue  # Only benchmark failures that have not happened yet.
            velocities = []
            for bid in env.top_ids:
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, int(bid), vel, 0)
                velocities.append(vel[3:].copy())
            # Score future failure, not a label inferred from image brightness.
            env.step(1.0)
            label = env.failed()
            split = "train" if i % 10 < 6 else "validation" if i % 10 < 8 else "test"
            file = f"episode_{episode_seed}.npz"
            np.savez_compressed(output / file, clip=clip, risky=np.bool_(label), seed=episode_seed,
                                velocity=np.mean(velocities, axis=0).astype(np.float32))
            records.append(dict(file=file, seed=episode_seed, split=split, risky=label,
                                failed_during_observation=failed_before, force_n=force))
            if i % 10 == 0:
                print(json.dumps(dict(episode=i + 1, risky=label, force_n=force)), flush=True)
        finally:
            env.close()
    (output / "manifest.json").write_text(json.dumps(dict(backend="mujoco", version=mujoco.__version__,
        task="predict tower failure within 1s after a short force disturbance", frames=4,
        frame_interval_s=.02, size=224, rejected_already_failed=attempt-episodes,
        records=records), indent=2))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/mujoco_jenga")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()
    if args.viewer:
        import mujoco.viewer
        import time
        env = JengaPhysics(args.seed)
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    env.step(.02)
                    viewer.sync()
                    time.sleep(.02)
        finally:
            env.close()
    else:
        if args.episodes < 10:
            p.error("Use at least 10 episodes to populate all three splits")
        generate(args.output, args.episodes, args.seed)


if __name__ == "__main__":
    main()
