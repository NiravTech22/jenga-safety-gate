"""Turn fitted params into things Isaac Lab / the twin can use."""
import json

def isaac_lab_snippet(names, params, mjcf_to_usd_joint=None):
    mp = mjcf_to_usd_joint or {n: n for n in names}
    lines = ["from isaaclab.actuators import ImplicitActuatorCfg", "actuators = {"]
    for n, p in zip(names, params):
        lines.append(f'    "{n}": ImplicitActuatorCfg(joint_names_expr=["{mp[n]}"], stiffness={p.kp:.4g}, damping={p.kd:.4g}, '
                     f'armature={p.armature:.4g}, friction={p.frictionloss:.4g}, effort_limit_sim={p.tau_max:.4g}),')
    lines.append("}")
    lines.append("# Isaac has no native delay/backlash: wrap policy actions with sage_so101.servo-style pipeline:")
    lines.append("# " + json.dumps({n: {"delay_s": round(p.delay, 4), "backlash_rad": round(p.backlash, 4)} for n, p in zip(names, params)}))
    return "\n".join(lines)
