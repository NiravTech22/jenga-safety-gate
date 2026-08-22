#!/usr/bin/env python3
"""SO-101 arm and a Jenga tower in Isaac Sim 5.1, headless.

This is a physics check, not part of the PLAN.md pipeline. The pipeline's own
backend (src/sim) is a static-stability model chosen because the experiment
turns on stability margin rather than contact dynamics. This script exists to
answer a different question: does a 54-block stack of thin boxes actually
stand up under PhysX with the SO-101's real kinematics next to it, and can
this machine run that at all.

Run:
  ~/isaacsim/python.sh scripts/isaac/so101_jenga_scene.py
  ~/isaacsim/python.sh scripts/isaac/so101_jenga_scene.py --layers 18 --settle 400

Machine note: this host is below NVIDIA's stated minimum (8.59 GB VRAM vs 10,
16.64 GB RAM vs 32). Headless with GPU dynamics disabled is the configuration
most likely to fit. Rendering is off by default for the same reason.

STATUS ON THIS HOST: NOT YET EXECUTED. Isaac Sim 5.1.0 segfaults at startup
here, inside librtx.scenedb.plugin.so during carbOnPluginStartup, before any
user code runs. A bare SimulationApp({"headless": True}) crashes identically,
so this is environmental rather than a fault in this scene. Ruled out by
testing: the extension cache (was corrupt, repaired), the shader/texture/Kit
caches (cleared), every bundled Kit experience, multi-GPU selection, and
Vulkan ICD conflicts. Vulkan itself is healthy and the driver stack is
self-consistent at 595.84.

The two remaining candidates both need root and a reboot, so they are the
owner's call:
  1. IOMMU is enabled. NVIDIA flags this as a known cause of Omniverse
     instability on Linux; add amd_iommu=off to GRUB and reboot.
  2. Driver 595.84 (June 2026) is newer than Isaac Sim 5.1.0 was built
     against. Dropping to a production-branch driver is the usual fix.

The code below follows the official URDF-import sample shipped in
standalone_examples/api/isaacsim.asset.importer.urdf, so it should be correct,
but treat it as unverified until it has actually run.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = REPO_ROOT / "data" / "assets" / "so101" / "so101_new_calib.urdf"

# Standard Jenga block, metres. Matches src/sim/tower.py so results here are
# comparable with the pipeline's own backend.
BLOCK_L, BLOCK_W, BLOCK_H = 0.075, 0.025, 0.015
SLOTS = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--layers", type=int, default=18, help="18 = full tower")
    p.add_argument("--settle", type=int, default=300, help="physics steps to settle")
    p.add_argument("--physics-hz", type=float, default=240.0)
    p.add_argument("--render", action="store_true",
                   help="Enable the renderer and capture a frame. Costs VRAM.")
    p.add_argument("--tower-x", type=float, default=0.22,
                   help="Tower centre distance from the arm base, metres.")
    p.add_argument("--save-usd", type=Path, default=None)
    return p.parse_args()


args = parse_args()

# SimulationApp must be constructed before any omni/isaacsim import.
from isaacsim import SimulationApp                                    # noqa: E402

# "RaytracedLighting" is the only spelling Kit accepts; a typo here is not
# validated and takes the RTX plugin down at startup.
_app_config = {
    "headless": True,
    "renderer": "RaytracedLighting",
    "enable_livestream": False,
    "disable_viewport_updates": not args.render,
}
simulation_app = SimulationApp(_app_config)

import numpy as np                                                    # noqa: E402
import omni.kit.commands                                              # noqa: E402
import omni.usd                                                       # noqa: E402
from isaacsim.core.api import World                                   # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid                   # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial               # noqa: E402
from isaacsim.core.prims import Articulation                          # noqa: E402
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdLux, UsdPhysics  # noqa: E402


def build_physics_scene(stage, physics_hz: float):
    """Solver settings tuned for a tall stack of small, thin boxes.

    Defaults are wrong for this: with 54 bodies at 15 mm thickness the stack
    jitters itself apart. The changes that matter are a high position-iteration
    count, TGS, contact offsets scaled to the block size rather than to a
    metre-scale robot, and CCD so a fast gripper cannot tunnel through a block.
    """
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    api = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    api.CreateEnableCCDAttr(True)
    api.CreateEnableStabilizationAttr(True)
    # GPU dynamics off: this host has 8.59 GB VRAM against a 10 GB minimum,
    # and a 54-box scene is well within what CPU PhysX handles.
    api.CreateEnableGPUDynamicsAttr(False)
    api.CreateBroadphaseTypeAttr("MBP")
    api.CreateSolverTypeAttr("TGS")
    api.CreateTimeStepsPerSecondAttr(physics_hz)
    return scene


def build_tower(world, n_layers: int, centre_x: float, material):
    """54 rigid boxes in the standard alternating pattern."""
    blocks = []
    for layer in range(n_layers):
        z = BLOCK_H * (layer + 0.5)
        along_x = (layer % 2 == 0)
        for slot in range(SLOTS):
            offset = (slot - 1) * BLOCK_W
            if along_x:
                pos = (centre_x, offset, z)
                scale = (BLOCK_L, BLOCK_W, BLOCK_H)
            else:
                pos = (centre_x + offset, 0.0, z)
                scale = (BLOCK_W, BLOCK_L, BLOCK_H)

            block = world.scene.add(DynamicCuboid(
                prim_path=f"/World/tower/layer_{layer:02d}/block_{slot}",
                name=f"block_{layer:02d}_{slot}",
                position=np.array(pos),
                scale=np.array(scale),
                color=np.array([0.82, 0.64, 0.44]),
                mass=0.02,                      # ~20 g, a real Jenga block
            ))
            block.apply_physics_material(material)

            prim = block.prim
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateSolverPositionIterationCountAttr(24)
            rb.CreateSolverVelocityIterationCountAttr(4)
            # Offsets scaled to a 15 mm block; metre-scale defaults leave
            # visible gaps and let the stack rock.
            rb.CreateContactSlopCoefficientAttr(0.0)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.0008)
            collision.CreateRestOffsetAttr(0.0)
            blocks.append(block)
    return blocks


def main():
    if not args.urdf.exists():
        print(f"URDF not found: {args.urdf}\n"
              f"Fetch it from https://github.com/TheRobotStudio/SO-ARM100 "
              f"(Simulation/SO101) into data/assets/so101/.")
        simulation_app.close()
        sys.exit(1)

    world = World(stage_units_in_meters=1.0,
                  physics_dt=1.0 / args.physics_hz,
                  rendering_dt=1.0 / 60.0)
    stage = omni.usd.get_context().get_stage()
    build_physics_scene(stage, args.physics_hz)

    PhysicsSchemaTools.addGroundPlane(stage, "/groundPlane", "Z", 10.0,
                                      Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.4))
    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/light"))
    light.CreateIntensityAttr(2500.0)

    # Wood on wood: friction is what holds a Jenga tower up, and getting it
    # wrong is the difference between a stack and a landslide.
    material = PhysicsMaterial(
        prim_path="/World/physics_materials/wood",
        static_friction=0.7, dynamic_friction=0.6, restitution=0.02,
    )

    print(f"Importing SO-101 from {args.urdf}")
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True            # bolted to the table
    import_config.distance_scale = 1.0
    import_config.default_drive_strength = 1e4
    import_config.default_position_drive_damping = 1e3

    status, arm_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(args.urdf),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        print("URDF import FAILED")
        simulation_app.close()
        sys.exit(1)
    print(f"  imported at {arm_prim_path}")

    blocks = build_tower(world, args.layers, args.tower_x, material)
    print(f"Built tower: {len(blocks)} blocks, {args.layers} layers, "
          f"height {args.layers * BLOCK_H * 1000:.0f} mm")

    world.reset()
    arm = Articulation(arm_prim_path)
    arm.initialize()
    print(f"Arm DOF: {arm.num_dof}  names: {arm.dof_names}")

    start_z = np.array([b.get_world_pose()[0][2] for b in blocks])
    print(f"\nSettling {args.settle} steps at {args.physics_hz:.0f} Hz "
          f"({args.settle / args.physics_hz:.2f} s simulated)...")
    for i in range(args.settle):
        world.step(render=args.render)

    end_pos = np.array([b.get_world_pose()[0] for b in blocks])
    drift_z = np.abs(end_pos[:, 2] - start_z)
    lateral = np.linalg.norm(end_pos[:, :2] - np.array(
        [b.get_world_pose()[0][:2] for b in blocks]), axis=1)
    top_z = end_pos[:, 2].max()
    expected_top = args.layers * BLOCK_H - BLOCK_H / 2

    print("\n--- tower after settling ---")
    print(f"  max vertical drift : {drift_z.max() * 1000:7.2f} mm")
    print(f"  mean vertical drift: {drift_z.mean() * 1000:7.2f} mm")
    print(f"  top block height   : {top_z * 1000:7.2f} mm "
          f"(expected {expected_top * 1000:.2f} mm)")
    standing = abs(top_z - expected_top) < 0.01
    print(f"  verdict            : {'STANDING' if standing else 'COLLAPSED'}")

    joint_positions = arm.get_joint_positions()
    print(f"\n  arm joint positions: {np.round(joint_positions, 4)}")

    if args.save_usd:
        args.save_usd.parent.mkdir(parents=True, exist_ok=True)
        omni.usd.get_context().save_as_stage(str(args.save_usd))
        print(f"\nStage saved to {args.save_usd}")

    simulation_app.close()
    return 0 if standing else 2


if __name__ == "__main__":
    sys.exit(main())
