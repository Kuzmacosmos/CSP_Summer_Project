import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from datetime import datetime
import os
from core_calculations import *

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Times New Roman'

PLANE_NORMAL = np.array([0.0, 0.0, -1.0])   # receiver plane unit normal always downward
PLANE_Z = RECEIVER_POS[2]                   # 771.6 mm
EPS = 1e-12
MIRROR_RADIUS = 12.5

def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Attempted to normalise a near‑zero vector.")
    return v / n

def _plane_basis_from_normal(n):
    """Return orthonormal (u, v) spanning the plane with normal n."""
    n = _unit(n)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 \
        else np.array([0.0, 1.0, 0.0])
    u = a - np.dot(a, n) * n
    u = _unit(u)
    v = np.cross(n, u)
    return u, v

def _incoming_for_energy(center_3d, s_chief, use_point_source, led_distance_mm):
    """
    Energy-weighting incoming direction at a given mirror centre.
    - Sun:     parallel beam along s_chief
    - LED:     ray from LED position L to the mirror centre
    """
    if use_point_source:
        L = LED_TARGET_POINT - led_distance_mm * s_chief  # same construction as core
        return _unit(center_3d - L)
    else:
        return s_chief

def compute_grid_surface_hits(h_key, tilt_deg, datetime_str, lat, lon, tz_off,
                              use_point_source, led_distance_mm,
                              led_divergence_deg):
    """Compatibility wrapper that reuses the core implementation and preserves
    the original return keys expected by the core module.
    """
    core = compute_heliostat_data(
        h_key, tilt_deg, datetime_str, lat, lon, tz_off,
        use_point_source=use_point_source,
        led_distance_mm=led_distance_mm,
        led_divergence_deg=led_divergence_deg
    )
    return {
        "surface_hits": core["surface_hits"],  # filled (alias)
        "surface_hits_filled": core["surface_hits_filled"],
        "surface_hits_ring": core["surface_hits_ring"],
        "endpoints": np.array(core["hits"]),
        "mirror_final_positions": core["mirror_final_positions"],
        "mirror_target_normals": core["mirror_target_normals"],
        "mirror_reflection_dirs": core["mirror_reflection_dirs"],
    }


def plot_receiver_plane(selected, datetime_str, lat, lon, tz_off, tilts, output_file,
                        use_point_source, led_distance_mm, led_divergence_deg):
    dtobj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    datetime_str_dmy = dtobj.strftime("%d-%m-%Y %H:%M")

    plt.figure(figsize=(8, 6))
    for idx, h_key in enumerate(selected):
        data = compute_grid_surface_hits(
            h_key, tilts[idx], datetime_str, lat, lon, tz_off,
            use_point_source=use_point_source,
            led_distance_mm=led_distance_mm,
            led_divergence_deg=led_divergence_deg
        )
        color = HU_COLORS.get(h_key, None)
        surf_hits = np.array(data['surface_hits'])
        if surf_hits.size > 0:
            plt.scatter(surf_hits[:, 0], surf_hits[:, 1], s=10, alpha=0.04,
                        color=color, label=f'{h_key} Mirrors Hits')

    plt.scatter(0, 0, marker='+', s=100, color='black', label='Receiver Centre')
    plt.xlabel('X (mm)'); plt.ylabel('Y (mm)')
    leg = plt.legend(loc='upper right', fontsize='small', ncol=2)
    for lh in leg.legend_handles: lh.set_alpha(1.0)

    plt.xlim(-150, 150); plt.xticks(np.arange(-150, 151, 50))
    plt.ylim(-100, 100); plt.yticks(np.arange(-100, 101, 20))
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.show()


##################################
# Centroid offset and spot spread
##################################

def compute_centroid(selected, datetime_str, lat, lon, tz_off, tilts,
                     use_point_source, led_distance_mm, led_divergence_deg):
    hits_list = []
    for idx, h_key in enumerate(selected):
        data = compute_grid_surface_hits(
            h_key, tilts[idx], datetime_str, lat, lon, tz_off,
            use_point_source=use_point_source,
            led_distance_mm=led_distance_mm,
            led_divergence_deg=led_divergence_deg
        )
        if data['surface_hits'].size:
            hits_list.append(data['surface_hits'])

    if not hits_list:
        return np.array([np.nan, np.nan, RECEIVER_POS[2]]), np.nan
    hits = np.vstack(hits_list)
    centroid = hits.mean(axis=0)
    mean_dist = np.linalg.norm(hits - centroid, axis=1).mean()
    return centroid, mean_dist

###########################################################
# Bright spot area and predicted projected pixel intensity
###########################################################

def ellipse_on_receiver(C, N, r_hat, R_mm, boundary_pts=300):
    """
    From mirror center C, normal N, reflected ray r_hat (unit), radius R_mm:
      - Intersect central ray with receiver plane (z = PLANE_Z).
      - Build ellipse principal vectors a_vec, b_vec in the plane.
      - Return (OK, center, a_vec, b_vec, area_mm2, polygon_xy)
    """
    r_z = r_hat[2]
    if abs(r_z) < EPS:
        return False, None, None, None, 0.0, None
    t = (PLANE_Z - C[2]) / r_z
    if t <= 0:
        return False, None, None, None, 0.0, None

    center = C + t * r_hat

    # Basis in mirror plane
    u, v = _plane_basis_from_normal(N)

    # Oblique projection of circle -> ellipse along r_hat onto receiver plane
    r_dot_np = float(np.dot(r_hat, PLANE_NORMAL))
    if abs(r_dot_np) < EPS:
        return False, None, None, None, 0.0, None

    u_dot_np = float(np.dot(u, PLANE_NORMAL))
    v_dot_np = float(np.dot(v, PLANE_NORMAL))
    a_vec = R_mm * (u - (u_dot_np / r_dot_np) * r_hat)
    b_vec = R_mm * (v - (v_dot_np / r_dot_np) * r_hat)

    # Analytic per-mirror area (exact for ellipse)
    # Equivalent form: area_mm2 = np.pi * np.linalg.norm(np.cross(a_vec, b_vec))
    area_mm2 = np.pi * R_mm ** 2 * (np.abs(np.dot(r_hat, N)) / np.abs(np.dot(
        r_hat, PLANE_NORMAL)))

    # Sample boundary once for Shapely polygon / union
    theta = np.linspace(0, 2*np.pi, boundary_pts, endpoint=False)
    pts3 = center[None, :] + np.cos(theta)[:, None]*a_vec + np.sin(theta)[:, None]*b_vec
    poly_xy = pts3[:, :2]  # drop z (constant on the plane)

    return True, center, a_vec, b_vec, float(area_mm2), poly_xy

def compute_projected_areas_shapely(
    selected, datetime_str, lat, lon, tz_off, tilts,
    mirror_radius_mm, boundary_pts,
    reflectance, atmos_loss_frac, frame_rect, use_point_source,
        led_distance_mm, led_divergence_deg):
    """
    Now also returns energy metrics weighted by rho = reflectance * (1 - atmos_loss_frac).
    If frame_rect is given, computes camera-matched E_roi (clipped to frame).
    """

    rho = reflectance * (1.0 - atmos_loss_frac)  # === 0.9801
    s_chief = _unit(compute_incoming(datetime_str, lat, lon, tz_off))

    per_mirror = []
    polygons = []
    mirror_normals = []
    refl_dirs = []
    cos_theta_list = []

    for idx, h_key in enumerate(selected):
        data = compute_grid_surface_hits(
            h_key, tilts[idx], datetime_str, lat, lon, tz_off,
            use_point_source,
            led_distance_mm,
            led_divergence_deg)
        C_all = data['mirror_final_positions']     # (2,3)
        N_all = data['mirror_target_normals']      # (2,3)
        r_all = data['mirror_reflection_dirs']     # (2,3), unit

        for m_idx, (C, N, r_hat) in enumerate(zip(C_all, N_all, r_all)):
            ok, center, a_vec, b_vec, area, poly_xy = ellipse_on_receiver(
                C, N, r_hat, mirror_radius_mm, boundary_pts=boundary_pts)
            if not ok or area <= 0:
                continue

            a1_hat = _incoming_for_energy(C, s_chief, use_point_source, led_distance_mm)
            cos_theta_list.append(abs(float(np.dot(a1_hat, N))))

            polygons.append(Polygon(poly_xy))
            mirror_normals.append(N)
            refl_dirs.append(r_hat)
            per_mirror.append({
                "heliostat": h_key,
                "mirror_index": m_idx,
                "area_mm2": area,
                "center_xy": (float(center[0]), float(center[1]))
            })

    # Areas (no overlaps vs union)
    sum_no_overlap = float(sum(p["area_mm2"] for p in per_mirror)) if per_mirror else 0.0
    union_area = float(unary_union(polygons).area) if polygons else 0.0

    # ENERGY METRICS
    # Best-case analytic total (all spots fully inside frame):
    A_m = np.pi * (mirror_radius_mm ** 2)
    E_theory = rho * A_m * float(np.sum(cos_theta_list))

    # ROI case clipping to camera frame and letting overlaps add
    E_roi = None
    if frame_rect is not None and polygons:
        from shapely.geometry import box
        xmin, ymin, xmax, ymax = frame_rect
        frame = box(xmin, ymin, xmax, ymax)

        n_p = np.array([0.0, 0.0, -1.0])  # receiver plane normal

        E_roi_val = 0.0
        for cos_theta, N_i, r_i, P_i in zip(cos_theta_list, mirror_normals, refl_dirs, polygons):
            num = abs(float(np.dot(n_p, r_i)))
            den = abs(float(np.dot(N_i, r_i))) + 1e-12
            I_ri = rho * cos_theta * (num / den)

            A_clip = float(P_i.intersection(frame).area)
            E_roi_val += I_ri * A_clip

        E_roi = E_roi_val

    return {
        "per_mirror": per_mirror,
        "sum_no_overlap_mm2": sum_no_overlap,
        "union_area_mm2": union_area,
        "rho": rho,
        "E_theory": E_theory,  # best-case analytic (all inside frame)
        "E_roi": E_roi         # camera-matched frame limits
    }

if __name__ == '__main__':
    # Source model control
    USE_POINT_SOURCE   = True      # True = LED point source; False = Sun (parallel)
    LED_DISTANCE_MM    = 9800      # finite source distance (mm)
    LED_DIVERGENCE_DEG = 6.5       # full divergence angle (deg)

    target = np.array([0, 0, 771.6])  # DO NOT CHANGE!
    frame_rect = (-150.0, -100.0, 150.0, 100.0) # DO NOT CHANGE!
    k_px = 2370.8 # Changeable. Scaling factor for projected pixel intensity

    selected = ['H1', 'H2', 'H3', 'H4']
    tilts = [10.5, 9, 11.6, 9]
    datetime_str = '2025-12-21 13:00:00'
    lat, lon, tz_off = 51.49, -0.177, 0

    fig_out_base = 'LED_receiver_plane_with_samples.png' if USE_POINT_SOURCE \
        else 'Receiver_plane_samples.png'
    ts = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
    out_name = (f"../Result/code_test"
                f"/{os.path.splitext(fig_out_base)[0]}_{ts}.png")
    log_txt = (f"../Result/code_test/LED_log_{ts}.txt") if USE_POINT_SOURCE \
        else (f"../Result/code_test/log_{ts}.txt")
    os.makedirs(os.path.dirname(log_txt), exist_ok=True)

    incoming = compute_incoming(datetime_str, lat, lon, tz_off)
    incoming /= np.linalg.norm(incoming)
    centroid, spot_spread = compute_centroid(
        selected, datetime_str, lat, lon, tz_off, tilts,
        use_point_source=USE_POINT_SOURCE,
        led_distance_mm=LED_DISTANCE_MM,
        led_divergence_deg=LED_DIVERGENCE_DEG
    )

    area_intensity_results = compute_projected_areas_shapely(
        selected, datetime_str, lat, lon, tz_off, tilts,
        mirror_radius_mm=12.5, boundary_pts=300,
        reflectance=0.99, atmos_loss_frac=0.01, frame_rect=frame_rect,
        use_point_source=USE_POINT_SOURCE,
        led_distance_mm=LED_DISTANCE_MM,
        led_divergence_deg=LED_DIVERGENCE_DEG
    )

    per_normals=[]; per_positions=[]; per_refl_dirs=[]
    for idx, h_key in enumerate(selected):
        data = compute_grid_surface_hits(
            h_key, tilts[idx], datetime_str, lat, lon, tz_off,
            use_point_source=USE_POINT_SOURCE,
            led_distance_mm=LED_DISTANCE_MM,
            led_divergence_deg=LED_DIVERGENCE_DEG
        )
        per_normals.append(data['mirror_target_normals'])
        per_positions.append(data['mirror_final_positions'])
        per_refl_dirs.append(data['mirror_reflection_dirs'])

    # Logging
    with open(log_txt, 'w') as txtf:
        print(f"At {datetime_str} local time, ({lat}, {lon})\n", file=txtf)

        print(f"----- Simulation configurations -----", file=txtf)
        print(f"Using point-source model: {USE_POINT_SOURCE}, LED distance (mm):"
              f" {LED_DISTANCE_MM}, full divergence (deg): {LED_DIVERGENCE_DEG}", file=txtf)
        print("Normalised chief-axis vector (s_hat):", incoming, file=txtf)
        print("Tilts (degrees):", tilts, file=txtf)

        print(f"\n----- Performance Metrics -----", file=txtf)
        print(f"Global centroid coordinate (mm): {centroid}", file=txtf)
        print(f"Centroid offset (mm): {np.linalg.norm(target - centroid)}",
              file=txtf)
        print(f"Spot spread (mm): {spot_spread}", file=txtf)
        print(f"Sum bright spot areas (no overlap, mm^2):"
              f" {area_intensity_results['sum_no_overlap_mm2']:.2f}",
              file=txtf)
        print(f"Union bright spot area (with overlap, mm^2):"
              f" {area_intensity_results['union_area_mm2']:.2f}",
              file=txtf)
        print(f"Pixel intensity scaling factor: {k_px}", file=txtf)
        print(f"Preliminary optical transmission factor ="
              f" {area_intensity_results['rho']:.4f}", file=txtf)
        print(f"Predicted pixel intensity (all inside, a.u.): "
              f" {k_px * area_intensity_results['E_theory']:.6g}",
              file=txtf)
        if area_intensity_results['E_roi'] is not None:
            print(f"Predicted pixel intensity (clipped to frame, a.u.): "
                f"{k_px * area_intensity_results['E_roi']:.6g}", file=txtf)

        print(f"\n----- Per-mirror log -----", file=txtf)
        for h_key, norms, poss, dirs in zip(selected, per_normals, per_positions, per_refl_dirs):
            print(f"{h_key} mirrors target normals:", norms, file=txtf)
            print(f"{h_key} mirrors target positions:", poss, file=txtf)
            print(f"{h_key} mirrors reflection unit rays:", dirs, file=txtf)


    plot_receiver_plane(
        selected, datetime_str, lat, lon, tz_off, tilts, output_file=out_name,
        use_point_source=USE_POINT_SOURCE,
        led_distance_mm=LED_DISTANCE_MM,
        led_divergence_deg=LED_DIVERGENCE_DEG
    )