import numpy as np
import math

def quat_conj(q):
    """Quaternion conjugate [w, x, y, z] -> [w, -x, -y, -z]."""
    w, x, y, z = q
    return np.array([ w, -x, -y, -z ])

def quat_mul(a, b):
    """Hamilton product a * b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_to_axis_angle(q):
    """Normalize q then return (axis, angle_rad)."""
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    angle = 2 * math.acos(np.clip(w, -1.0, 1.0))
    s = math.sqrt(max(0.0, 1 - w*w))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return xyz / s, angle

def axis_angle_to_quat(axis, angle_rad):
    """Convert (axis, angle) back into [w, x, y, z]."""
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([ math.cos(half), *(axis * math.sin(half)) ])

def compute_alignment(obs, STEPSIZE_DEG=5.0):
    """
    Given an observation dict, compute and print the full rotation
    needed to align and return a clamped quaternion step.
    """
    # 1) pull quaternions
    q_cur = np.array(obs["robot0_eef_quat_site"])   # [w, x, y, z]
    q_des = np.array(obs["Bread_quat"])        # [w, x, y, z]

    # 2) difference quaternion
    q_diff = quat_mul(q_des, quat_conj(q_cur))

    # 3) axis-angle of full rotation
    axis_full, angle_full = quat_to_axis_angle(q_diff)
    angle_deg = math.degrees(angle_full)
    print(f"Full rotation needed: {angle_deg:.2f}° about axis {axis_full}")

    # 4) clamp to step size
    step_rad = math.radians(min(abs(angle_deg), STEPSIZE_DEG))
    q_step = axis_angle_to_quat(axis_full, step_rad)

    return axis_full, angle_deg, q_step

def quat_to_axis_angle(q):
    """Normalize q then return (axis, angle_rad)."""
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    angle = 2 * math.acos(np.clip(w, -1.0, 1.0))
    s = math.sqrt(max(0.0, 1 - w*w))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return xyz / s, angle


# --- Example with your provided data ---
if __name__ == "__main__":
    obs1 = {
      "robot0_eef_quat_site": [
        0.7155203819274902,
          0.6933568716049194,
          0.06052280589938164,
          0.060197729617357254
      ],
      "Bread_quat": [
        0.21897956728935242,
          0.9720696806907654,
          0.020145511254668236,
          0.08199377357959747
      ]
    }
    obs2 = {
      "robot0_eef_quat_site": [
                 0.9963843875304679,
          0.0009752076517845487,
          0.08493359698192877,
          0.0018669116960239075
      ],
      "Bread_quat": [
       -0.35860174894332886,
          0.9308927059173584,
          -0.023990359157323837,
          0.06533017754554749
      ]
    }
    axis, full_deg, step_q = compute_alignment(obs1, STEPSIZE_DEG=2.0)

    axis, ang = quat_to_axis_angle(obs1["Bread_quat"])
    axis1, ang1 = quat_to_axis_angle(obs1["robot0_eef_quat_site"])

    axis1, full_deg1, step_q1 = compute_alignment(obs2, STEPSIZE_DEG=2.0)

    axis2, ang2= quat_to_axis_angle(obs2["Bread_quat"])
    axis3, ang3 = quat_to_axis_angle(obs2["robot0_eef_quat_site"])

    print(ang, math.degrees(ang))
    print(ang1, math.degrees(ang1))

    print(ang2, math.degrees(ang2))
    print(ang3, math.degrees(ang3))
    print("Clamped step quaternion:", step_q)