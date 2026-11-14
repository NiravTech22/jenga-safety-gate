#!/usr/bin/env python3
"""SO-101 + custom claw: one-scene teleoperation and physics pick/place baseline.

Builds twin/models/scene_pickplace.generated.xml from scene_custom_claw_physics.xml
each run. Nothing in the source models or on the hardware is modified.

Usage (from the repository root):
  python twin/so101_pickplace.py                    # MuJoCo window, auto-plays the attempt
  python twin/so101_pickplace.py --headless         # PASS/FAIL only
  python twin/so101_pickplace.py --record run.mp4   # offscreen video, no window
  python twin/so101_pickplace.py --manual           # window + keyboard teleop, no auto-play

Needs the upstream SO-101 meshes once: python scripts/tools/fetch_so101_assets.py

Controls in MuJoCo window:
  P: scripted pick/place; W/S: world X +/-; A/D: world Y +/-;
  R/F: world Z +/-; Q/E: wrist roll +/-; T/G: wrist flex +/-;
  O: open gripper; C: close gripper; H: reset scene.

The robot has five arm DOF, so arbitrary simultaneous 6-DOF Cartesian
position and orientation is physically underactuated. XYZ uses damped least
squares IK with a soft orientation constraint; Q/E and T/G rotate via the real
wrist actuators.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

TWIN = Path(__file__).resolve().parent
ROOT = TWIN / 'models'
SOURCE = ROOT / 'scene_custom_claw_physics.xml'
OUT = ROOT / 'scene_pickplace.generated.xml'          # rebuilt every run; gitignored
RUN_LOG = TWIN.parent / 'logs/twin/pickplace_run.json'

ARM = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
SEED_Q = np.array([-0.00536, -0.01648, -0.20716,  1.6071 ,  0.04148])
PICK_XY = np.array([0.26321, 0.00000])
PLACE_XY = PICK_XY + np.array([-0.06, 0.0])
TABLE_HALF = np.array([0.02, 0.025, 0.008])
TABLE_TOP_CANDIDATES = [0.04]
CUBE_HALF = .0125
GRASP_WIDTH = .075
CUBE_MASS = .020
OPEN = 0.0
PAD_NAMES = ('pad_finger_left', 'pad_finger_right')


def fmt(xs):
    return ' '.join(f'{float(x):.9g}' for x in xs)


def ensure_scene():
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Missing working physics scene: {SOURCE}')
    missing = [m for m in ('base_so101_v2.stl', 'upper_arm_so101_v1.stl')
               if not (ROOT / 'meshes' / m).is_file()]
    if missing:
        raise SystemExit('Upstream SO-101 meshes are missing. Run once from the repo root:\n'
                         '  python scripts/tools/fetch_so101_assets.py')
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    world = root.find('worldbody')
    if world is None:
        raise RuntimeError('Physics scene missing <worldbody>')
    for node in list(world):
        name = node.get('name', '')
        if name in ('demo_table', 'grasp_cube') or name.startswith('workcell_leg_'):
            world.remove(node)
    # Two independently anchored platforms, rather than a table attached to
    # the arm's current fingertip coordinates. Both platforms collide normally.
    for name, xy in [('pick_table', PICK_XY), ('place_table', PLACE_XY)]:
        ET.SubElement(world, 'geom', {
            'name': name, 'type': 'box',
            'pos': fmt([xy[0], xy[1], .200 - TABLE_HALF[2]]),
            'size': fmt(TABLE_HALF), 'friction': '1.3 0.02 0.002',
            'contype': '1', 'conaffinity': '1', 'rgba': '.48 .50 .55 1',
        })
        # Fixture posts are visual-only at this stage. The top platforms
        # themselves are physical collision geoms, not visual placeholders.
        ET.SubElement(world, 'geom', {
            'name': f'{name}_post', 'type': 'cylinder',
            'pos': fmt([xy[0], xy[1], .100 - TABLE_HALF[2]]),
            'size': fmt([.007, .100 - TABLE_HALF[2]]),
            'rgba': '.31 .33 .35 1', 'contype': '0', 'conaffinity': '0',
        })
    cube = ET.SubElement(world, 'body', {
        'name': 'grasp_cube',
        'pos': fmt([*PICK_XY, .200 + CUBE_HALF + .0002]),
    })
    ET.SubElement(cube, 'freejoint', {'name': 'grasp_cube_free'})
    ET.SubElement(cube, 'geom', {
        'name': 'grasp_cube_geom', 'type': 'box',
        'size': fmt([CUBE_HALF, GRASP_WIDTH/2, CUBE_HALF]), 'mass': str(CUBE_MASS),
        'friction': '1.9 .03 .003', 'condim': '4',
        'solref': '.01 1', 'contype': '1', 'conaffinity': '1',
        'rgba': '.14 .43 .94 1',
    })
    # Offscreen framebuffer large enough for --record.
    vis = root.find('visual') if root.find('visual') is not None else ET.SubElement(root, 'visual')
    glob = vis.find('global') if vis.find('global') is not None else ET.SubElement(vis, 'global')
    glob.set('offwidth', '1280'); glob.set('offheight', '960')
    # Two camera definitions are not required to operate the picker. They can
    # be introduced after the physical pick/place is repeatable for ACT data.
    ET.indent(tree, space='  ')
    tree.write(OUT, encoding='utf-8', xml_declaration=True)


def jadr(model, name):
    return int(model.jnt_qposadr[model.joint(name).id])


def init_state(model, data, top, q=SEED_Q):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.qpos0
    for name, val in zip(ARM, q):
        data.qpos[jadr(model, name)] = float(val)
        data.ctrl[model.actuator(name).id] = float(val)
    for finger in ('gripper', 'gripper_mimic'):
        data.qpos[jadr(model, finger)] = OPEN
    data.ctrl[model.actuator('gripper').id] = OPEN
    adr = jadr(model, 'grasp_cube_free')
    data.qpos[adr:adr+3] = [*PICK_XY, top+CUBE_HALF+.0002]
    data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    left_id, right_id = [
        model.geom(n).id for n in PAD_NAMES
    ]
    jaw_axis = (
        data.geom_xpos[right_id]
        - data.geom_xpos[left_id]
    )
    yaw = float(np.arctan2(-jaw_axis[0], jaw_axis[1]))

    data.qpos[adr+3:adr+7] = [
        np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
    ]

    mujoco.mj_forward(model, data)


def place_tables(model, top):
    for name, xy in [('pick_table', PICK_XY), ('place_table', PLACE_XY)]:
        gid = model.geom(name).id
        model.geom_pos[gid] = [xy[0], xy[1], top-TABLE_HALF[2]]
        post = model.geom(f'{name}_post').id
        # Post is just a visual fixture and never participates in contacts.
        model.geom_pos[post] = [xy[0], xy[1], (top-2*TABLE_HALF[2]) / 2]
        model.geom_size[post, :2] = [.007, max(.001, (top-2*TABLE_HALF[2])/2)]


def table_collision_problems(model, data, threshold=.003):
    platforms = {model.geom(name).id for name in ('pick_table','place_table')}
    cube = model.geom('grasp_cube_geom').id
    problems = []
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if pair & platforms and cube not in pair and float(c.dist) < -threshold:
            other = list(pair - platforms)
            if other:
                gid = other[0]
                bid = int(model.geom_bodyid[gid])
                other_name = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, gid
                    ) or f'unnamed_geom_{gid}'
                )
                body_name = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, bid
                    ) or f'unnamed_body_{bid}'
                )
                culprit = (
                    f'{other_name} [body={body_name}, '
                    f'gid={gid}, contact_xyz={np.round(c.pos, 4)}]'
                )
            else:
                culprit = 'overlapping_workcell_platforms'
            problems.append((culprit, float(c.dist)))
    return problems


def persist_selected_table_height(top):
    """Keep saved MJCF consistent with the selected in-memory geometry."""
    tree = ET.parse(OUT)
    world = tree.getroot().find('worldbody')
    for name, xy in [('pick_table', PICK_XY), ('place_table', PLACE_XY)]:
        geom = world.find(f"./geom[@name='{name}']")
        post = world.find(f"./geom[@name='{name}_post']")
        geom.set('pos', fmt([xy[0], xy[1], top-TABLE_HALF[2]]))
        post.set('pos', fmt([xy[0], xy[1], (top-2*TABLE_HALF[2])/2]))
        post.set('size', fmt([.007, max(.001,(top-2*TABLE_HALF[2])/2)]))
    cube = world.find("./body[@name='grasp_cube']")
    cube.set('pos', fmt([*PICK_XY, top+CUBE_HALF+.0002]))
    ET.indent(tree, space='  ')
    tree.write(OUT, encoding='utf-8', xml_declaration=True)


def workspace_setup():
    first_failure = None
    for top in TABLE_TOP_CANDIDATES:
        # Bake the platform height into the XML and compile it. Moving a static
        # world geom after compiling (model.geom_pos[...] = ...) leaves MuJoCo's
        # precomputed collision tree at the old height, so the block fell
        # straight through the platform without generating a single contact.
        ensure_scene()
        persist_selected_table_height(top)
        model = mujoco.MjModel.from_xml_path(str(OUT))
        data = mujoco.MjData(model)
        for name in ARM + ('gripper',):
            model.actuator(name)  # Fail clearly if an expected actuator is absent.
        for name in PAD_NAMES + ('pick_table', 'place_table', 'grasp_cube_geom'):
            model.geom(name)
        init_state(model, data, top)
        issues = table_collision_problems(model, data)
        if not issues:
            try:
                plan = plan_motion(model, data, top)
            except RuntimeError as e:
                print(f'Z={top:.3f} m: IK failed: {e}')
                continue

            # Check every planned arm pose against the physical tables.
            probe = mujoco.MjData(model)
            unsafe = None

            for phase, angles in plan.items():
                if phase in ('close_q', 'reference_axis'):
                    continue

                probe.qpos[:] = data.qpos
                probe.qvel[:] = 0

                for name, angle in zip(ARM, angles):
                    probe.qpos[jadr(model, name)] = float(angle)

                mujoco.mj_forward(model, probe)

                hits = table_collision_problems(model, probe)

                if hits:
                    unsafe = (phase, hits[:2])
                    break

            if unsafe:
                print(
                    f'Z={top:.3f} m: '
                    f'waypoint-table collision: {unsafe}'
                )
                continue

            print(f'Initial collision-free hover: platform top Z={top:.3f} m')
            persist_selected_table_height(top)
            return model, data, top
        if first_failure is None:
            first_failure = issues[:3]
    raise RuntimeError(
        'No tested tabletop height passed IK and static clearance checks. '
        f'Example contacts: {first_failure}. The scene was not validated.'
    )


def xyz_and_rot(model, data):
    ids = [model.geom(n).id for n in PAD_NAMES]
    p = .5 * (data.geom_xpos[ids[0]] + data.geom_xpos[ids[1]])
    R = data.xmat[model.body('custom_gripper_base').id].reshape(3,3)
    return np.array(p, dtype=float).copy(), R.copy()


def tool_axis(model, data):
    """World-space approach axis from the gripper base to pad midpoint."""
    left, right = [model.geom(n).id for n in PAD_NAMES]
    midpoint = 0.5 * (data.geom_xpos[left] + data.geom_xpos[right])
    base = data.xpos[model.body('custom_gripper_base').id]
    axis = midpoint - base
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise RuntimeError('Degenerate gripper tool axis')
    return axis / norm



def solve_ik(model, target, reference_axis, initial, gripper_q=OPEN,
             pos_tol=.002, max_iter=400):
    """Stable 5-DOF damped IK with adaptive damping and line search."""

    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    d.qvel[:] = 0

    for name in ('gripper', 'gripper_mimic'):
        d.qpos[jadr(model, name)] = gripper_q

    q = np.asarray(initial, dtype=float).copy()
    qpos = [jadr(model, n) for n in ARM]
    dof = [int(model.jnt_dofadr[model.joint(n).id]) for n in ARM]

    limits = []
    for name in ARM:
        jid = model.joint(name).id
        if model.jnt_limited[jid]:
            low, high = map(float, model.jnt_range[jid])
        else:
            low, high = -3.0, 3.0
        limits.append((low + .025, high - .025))

    lp, rp = [model.geom(n).id for n in PAD_NAMES]
    body = model.body('custom_gripper_base').id

    jacpL = np.zeros((3, model.nv))
    jacpR = np.zeros((3, model.nv))
    jacpB = np.zeros((3, model.nv))

    ref_axis = np.asarray(reference_axis, dtype=float).copy()
    ref_axis /= max(float(np.linalg.norm(ref_axis)), 1e-12)

    eye = np.eye(3)
    weight = .075
    damping = .02

    target = np.asarray(target, dtype=float)

    def evaluate(qv):
        for i, adr in enumerate(qpos):
            d.qpos[adr] = qv[i]

        mujoco.mj_forward(model, d)

        current, _ = xyz_and_rot(model, d)
        base_pos = d.xpos[body].copy()

        v = current - base_pos
        vlen = float(np.linalg.norm(v))

        if vlen < 1e-9:
            return 1e9, current, np.zeros(3), np.zeros(3), np.pi

        axis = v / vlen
        ep = target - current
        ea = ref_axis - axis

        angle = float(np.arccos(
            np.clip(np.dot(axis, ref_axis), -1.0, 1.0)
        ))

        cost = float(ep @ ep + weight**2 * (ea @ ea))

        return cost, current, axis, ep, angle

    for iteration in range(max_iter):

        cost, current, axis, ep, angle = evaluate(q)

        if np.linalg.norm(ep) < pos_tol and angle < .20:
            return q.copy(), float(np.linalg.norm(ep)), angle

        mujoco.mj_jacGeom(model, d, jacpL, None, lp)
        mujoco.mj_jacGeom(model, d, jacpR, None, rp)
        mujoco.mj_jacBody(model, d, jacpB, None, body)

        Jmid = .5 * (
            jacpL[:, dof] + jacpR[:, dof]
        )

        Jv = Jmid - jacpB[:, dof]

        Jaxis = (
            (eye - np.outer(axis, axis)) @ Jv
        ) / max(float(np.linalg.norm(current -
                                     d.xpos[body])), 1e-9)

        J = np.vstack((
            Jmid,
            weight * Jaxis
        ))

        error = np.concatenate((
            ep,
            weight * (ref_axis - axis)
        ))

        H = J.T @ J + damping**2 * np.eye(len(ARM))

        try:
            dq = np.linalg.solve(H, J.T @ error)
        except np.linalg.LinAlgError:
            damping *= 3
            continue

        max_step = float(np.max(np.abs(dq)))

        if max_step > .025:
            dq *= .025 / max_step

        accepted = False

        for scale in (1.0, .5, .25, .125, .0625, .03125):

            trial = q + scale * dq

            for i, (lo, hi) in enumerate(limits):
                trial[i] = np.clip(trial[i], lo, hi)

            trial_cost, _, _, _, _ = evaluate(trial)

            if trial_cost < cost - 1e-12:
                q = trial.copy()
                damping = max(.003, damping * .75)
                accepted = True
                break

        if not accepted:
            damping = min(.5, damping * 3)

            if damping >= .5:
                break

    _, current, axis, ep, angle = evaluate(q)

    return q.copy(), float(np.linalg.norm(ep)), angle


def jaw_close_target(model, data, cube_side=GRASP_WIDTH):
    """Gripper: q=0 open (110mm); q=.03 closed (50mm)."""
    tmp = mujoco.MjData(model)
    tmp.qpos[:] = data.qpos
    tmp.qvel[:] = 0

    left, right = [model.geom(n).id for n in PAD_NAMES]
    gaps = []

    for stroke in (0.0, 0.03):
        for joint in ('gripper', 'gripper_mimic'):
            tmp.qpos[jadr(model, joint)] = stroke

        mujoco.mj_forward(model, tmp)

        separation = np.linalg.norm(
            tmp.geom_xpos[right] - tmp.geom_xpos[left]
        )

        gaps.append(float(separation - 0.004))

    open_gap, closed_gap = gaps

    if not (closed_gap + .005 < cube_side < open_gap - .005):
        raise RuntimeError(
            f"Invalid grasp: open={open_gap*1000:.1f}mm, "
            f"closed={closed_gap*1000:.1f}mm, "
            f"object={cube_side*1000:.1f}mm"
        )

    target_gap = cube_side - .016  # ~8 mm squeeze per jaw pair: enough normal force that the block can't slip

    stroke = .03 * (
        (open_gap - target_gap) /
        (open_gap - closed_gap)
    )

    return float(np.clip(stroke, 0, .03)), (
        closed_gap, open_gap
    )


def plan_motion(model, data, top):
    _, _ = xyz_and_rot(model, data)
    reference_axis = tool_axis(model, data)
    cube = np.array([*PICK_XY, top + CUBE_HALF])
    place = np.array([*PLACE_XY, top + CUBE_HALF])
    goals = [
        ('pregrasp', cube + [0, 0, 0.045]),
        ('approach_mid', cube + [0, 0, .040]),
        ('approach_low', cube + [0, 0, .020]),
        ('approach', cube + [0, 0, .003]),
        ('lift', cube + [0, 0, 0.045]),
        ('transfer', place + [0, 0, .045]),
        ('transfer_mid', place + [0, 0, 0.045]),
        ('transfer_low', place + [0, 0, .040]),
        ('lower_mid', place + [0, 0, .020]),
        ('lower', place + [0, 0, .003]),
        ('retreat', place + [0, 0, 0.045]),
    ]
    result = {}
    q_prev = SEED_Q.copy()
    for name, target in goals:
        q, pe, ae = solve_ik(model, target, reference_axis, q_prev)
        print(f'IK {name:12s} position_error={pe*1000:5.1f}mm '
              f'approach_axis_error={np.degrees(ae):4.1f}deg')
        if pe > .012 or ae > .35:
            raise RuntimeError(
                f'IK cannot reach {name}: position error {pe*1000:.1f}mm, '
                f'approach-axis error {np.degrees(ae):.1f}deg. '
                'The Cartesian target is outside the current 5-DOF reachable set.'
            )
        result[name] = q.copy()
        q_prev = q
    close, (closed_gap, open_gap) = jaw_close_target(model, data)
    print(f'Jaw opening range: {closed_gap*1000:.1f} to '
          f'{open_gap*1000:.1f}mm; block_width=75mm; close target={close*1000:.1f}mm')
    result['close_q'] = close
    result['reference_axis'] = reference_axis.copy()
    return result


def contact_counts(model,data):
    cube = model.geom('grasp_cube_geom').id
    pads = [model.geom(n).id for n in PAD_NAMES]
    counts = [0,0]
    for k in range(data.ncon):
        pair = {int(data.contact[k].geom1), int(data.contact[k].geom2)}
        for i,p in enumerate(pads):
            if pair == {cube,p}:
                counts[i]+=1
    return counts


def perform_auto(model,data,top,waypoints,viewer=None, abort=lambda: False):
    arm_acts = [model.actuator(n).id for n in ARM]
    grip_act = model.actuator('gripper').id
    cube_id = model.body('grasp_cube').id
    phases = [
        ('pregrasp', 'pregrasp', OPEN, 2.8),
        ('approach_mid', 'approach_mid', OPEN, 1.2),
        ('approach_low', 'approach_low', OPEN, 1.2),
        ('approach', 'approach', OPEN, 1.2),
        ('close', 'approach', waypoints['close_q'], 1.4),
        ('hold', 'approach', waypoints['close_q'], 1.2),
        ('lift', 'lift', waypoints['close_q'], 3.0),
        ('transfer', 'transfer', waypoints['close_q'], 3.6),
        ('transfer_mid', 'transfer_mid', waypoints['close_q'], 1.0),
        ('transfer_low', 'transfer_low', waypoints['close_q'], 1.0),
        ('lower_mid', 'lower_mid', waypoints['close_q'], 1.0),
        ('lower', 'lower', waypoints['close_q'], 1.0),
        ('release', 'lower', OPEN, 1.2),
        ('retreat', 'retreat', OPEN, 2.8),
        ('settle', 'retreat', OPEN, 1.2),
    ]
    logs=[]
    total_contact_steps=0
    for phase, goal, grip, secs in phases:
        if abort():
            print('Auto sequence interrupted.')
            break
        q_start = np.asarray(data.ctrl[arm_acts]).copy()
        q_end = waypoints[goal]
        grip_start = float(data.ctrl[grip_act])
        nsteps = max(1,round(secs/model.opt.timestep))
        both=0
        start_time=time.monotonic()
        for i in range(nsteps):
            if abort(): break
            u = (i+1)/nsteps
            # Quintic blend, smooth zero-velocity and zero-acceleration ends.
            s = 6*u**5-15*u**4+10*u**3
            data.ctrl[arm_acts] = q_start + s*(q_end-q_start)
            data.ctrl[grip_act] = grip_start + s*(grip-grip_start)
            mujoco.mj_step(model,data)
            if min(contact_counts(model,data)) > 0:
                both += 1
            if viewer is not None and i%20==0:
                viewer.sync()
                if getattr(viewer, 'realtime', True):
                    # Roughly wall-clock simulation speed in the live viewer.
                    dt = (i+1)*model.opt.timestep-(time.monotonic()-start_time)
                    if dt > 0: time.sleep(min(dt,.035))
        mujoco.mj_forward(model,data)
        pos = data.xpos[cube_id].copy()
        pad_contacts = contact_counts(model,data)
        phase_rec = dict(phase=phase,cube_xyz=pos.tolist(),
                         pad_contacts=pad_contacts,both_contact_steps=both,
                         commanded_gripper=float(data.ctrl[grip_act]))
        logs.append(phase_rec)
        total_contact_steps += both
        print(f'{phase:9s} cube={np.round(pos,3)} '
              f'pads={pad_contacts} simultaneous_steps={both}',flush=True)
        if phase == 'hold' and both < .10*nsteps:
            print('STOP: cube was not held by both pads. No fake pickup.')
            break
        if phase == 'lift' and pos[2] < top+CUBE_HALF+.022:
            print('STOP: cube did not rise at least 22 mm. Not a successful lift.')
            break
        if phase in ('transfer', 'transfer_mid', 'transfer_low',
                      'lower_mid', 'lower') and min(contact_counts(model,data)) == 0:
            print('STOP: cube contact lost in transit. Not a successful place.')
            break
    place_xyz = data.xpos[cube_id].copy()
    placement_error = float(np.linalg.norm(place_xyz[:2]-PLACE_XY))
    # Require the cube to end on the PLACE fixture, not merely above it.
    cube_geom = model.geom('grasp_cube_geom').id
    place_geom = model.geom('place_table').id
    place_contacts = sum(
        {int(data.contact[k].geom1), int(data.contact[k].geom2)} == {cube_geom, place_geom}
        for k in range(data.ncon)
    )
    cube_free = model.joint('grasp_cube_free').id
    vel_adr = int(model.jnt_dofadr[cube_free])
    cube_speed = float(np.linalg.norm(data.qvel[vel_adr:vel_adr+3]))
    placed = bool(len(logs)==len(phases) and placement_error < .020
                  and abs(place_xyz[2]-(top+CUBE_HALF)) < .020
                  and place_contacts > 0 and cube_speed < .12)
    report = dict(success=placed,cube_final_xyz=place_xyz.tolist(),
                  place_table_contacts=place_contacts, cube_speed_mps=cube_speed,
                  placement_xy_error_m=placement_error,
                  both_contact_steps=total_contact_steps,
                  phases=logs,
                  observation='Scripted feedback controller, not a trained ACT policy')
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text(json.dumps(report,indent=2)+'\n')
    print(f'PICK AND PLACE: {"PASS" if placed else "FAIL"}; '
          f'XY error {placement_error*1000:.1f} mm; log={RUN_LOG}')
    return report


class VideoRecorder:
    """Offscreen camera that perform_auto() drives like a viewer (one frame per sync)."""
    realtime = False

    def __init__(self, model, data, path, width=960, height=720):
        self.model, self.data, self.path = model, data, Path(path)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [.235, 0., .07]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = .55, 135, -22
        self.frames = []

    def sync(self):
        self.renderer.update_scene(self.data, self.cam)
        self.frames.append(self.renderer.render().copy())

    def save(self, fps=25):
        try:
            import imageio.v2 as imageio
            imageio.mimwrite(self.path, self.frames, fps=fps, quality=8)
            print(f'Video saved: {self.path} ({len(self.frames)} frames)')
        except Exception as e:  # imageio/ffmpeg not installed: fall back to PNG frames
            from PIL import Image
            out = self.path.with_suffix('')
            out.mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(self.frames):
                Image.fromarray(f).save(out / f'frame_{k:04d}.png')
            print(f'Could not write video ({e}); saved {len(self.frames)} PNG frames to {out}/')


def run(headless=False, record=None, manual=False):
    model,data,top=workspace_setup()
    print('Compiles:',model.njnt,'joints;',model.nu,'actuators')
    # Plan before starting the simulation. Fail safely if target is unreachable.
    plan=None
    try:
        plan=plan_motion(model,data,top)
        print('All task-space waypoints are within initial IK tolerance.')
    except RuntimeError as e:
        print('AUTO PLAN NOT READY:',e)
        print('You can still move and rotate the arm interactively.')
    if headless or record:
        if plan is None:
            raise SystemExit('Cannot run pick/place: task-space preflight failed.')
        rec = VideoRecorder(model, data, record) if record else None
        perform_auto(model,data,top,plan,rec)
        if rec: rec.save()
        return

    import mujoco.viewer
    event_queue=deque()
    def on_key(keycode):
        try: event_queue.append(chr(keycode).upper())
        except (ValueError,OverflowError): pass

    print('\nViewer keys: W/S X, A/D Y, R/F Z | Q/E wrist roll, T/G wrist tilt')
    print('             O open, C close, P auto pick/place, H reset')
    targets=np.array([data.ctrl[model.actuator(n).id] for n in ARM])
    pos_goal,_=xyz_and_rot(model,data)
    axis_goal=tool_axis(model,data)
    arm_act=[model.actuator(n).id for n in ARM]
    grip_act=model.actuator('gripper').id
    with mujoco.viewer.launch_passive(model,data,key_callback=on_key) as viewer:
        viewer.cam.lookat[:]=[.235,0.,.07]
        viewer.cam.distance=.55
        viewer.cam.azimuth=135
        viewer.cam.elevation=-22
        if plan is not None and not manual:
            # Play the scripted attempt as soon as the window opens; press P to replay.
            print('Playing the pick/place attempt... (press P in the window to replay)')
            perform_auto(model,data,top,plan,viewer,abort=lambda: not viewer.is_running())
            targets=np.asarray(data.ctrl[arm_act]).copy()
            pos_goal,_=xyz_and_rot(model,data)
            axis_goal=tool_axis(model,data)
        while viewer.is_running():
            t0=time.monotonic()
            while event_queue:
                k=event_queue.popleft()
                if k=='H':
                    init_state(model,data,top)
                    targets=np.asarray(data.ctrl[arm_act]).copy()
                    pos_goal,_=xyz_and_rot(model,data)
                    axis_goal=tool_axis(model,data)
                elif k=='P':
                    if plan is None:
                        print('No collision-safe, reachable pick/place plan. Auto disabled.')
                    else:
                        init_state(model,data,top)
                        perform_auto(model,data,top,plan,viewer)
                        targets=np.asarray(data.ctrl[arm_act]).copy()
                        pos_goal,_=xyz_and_rot(model,data)
                        axis_goal=tool_axis(model,data)
                elif k in 'WSADRF':
                    delta={'W':( .005,0,0),'S':(-.005,0,0),
                           'A':(0, .005,0),'D':(0,-.005,0),
                           'R':(0,0, .005),'F':(0,0,-.005)}[k]
                    request=pos_goal+np.array(delta)
                    q,pe,re=solve_ik(model,request,axis_goal,targets)
                    if pe < .012:
                        targets=q
                        pos_goal=request
                    else:
                        print(f'IK rejected {k}: {pe*1000:.1f}mm position error')
                elif k in 'QETG':
                    idx=4 if k in 'QE' else 3
                    step=.07 if k in 'QT' else -.07
                    jid=model.joint(ARM[idx]).id
                    lo,hi=map(float,model.jnt_range[jid])
                    targets[idx]=np.clip(targets[idx]+step,lo+.02,hi-.02)
                    # Recompute reference orientation after direct wrist motion.
                    tmp=mujoco.MjData(model)
                    tmp.qpos[:]=data.qpos
                    for name,qi in zip(ARM,targets): tmp.qpos[jadr(model,name)]=qi
                    mujoco.mj_forward(model,tmp)
                    pos_goal,_=xyz_and_rot(model,tmp)
                    axis_goal=tool_axis(model,tmp)
                elif k=='O':data.ctrl[grip_act]=OPEN
                elif k=='C':
                    try:
                        close,_=jaw_close_target(model,data)
                        data.ctrl[grip_act]=close
                    except RuntimeError as e: print(e)
            data.ctrl[arm_act]=targets
            mujoco.mj_step(model,data)
            viewer.sync()
            dt=model.opt.timestep-(time.monotonic()-t0)
            if dt>0:time.sleep(dt)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--headless',action='store_true',help='Run scripted physics demo without GUI')
    ap.add_argument('--record',metavar='FILE.mp4',help='Render the attempt offscreen to a video (no window)')
    ap.add_argument('--manual',action='store_true',help='Open the viewer without auto-playing the attempt')
    args=ap.parse_args()
    run(headless=args.headless, record=args.record, manual=args.manual)

if __name__=='__main__':
    main()
