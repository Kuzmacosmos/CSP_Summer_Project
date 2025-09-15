import numpy as np
import datetime as dt

from sunPos      import solar_angles
from sph_to_cart import mod_sph_to_cart
from refl_ray    import refl_ray
from rot_helper  import rodrigues

# ----- Finite point source version -----
# ——— constants ———
RECEIVER_POS = np.array([0.0, 0.0, 771.6])
H_POSITIONS = {
    'H1': np.array([-222.5, -97.5, 199]),
    'H2': np.array([ 227.5, -97.5, 199]),
    'H3': np.array([-222.5,-597.5, 199]),
    'H4': np.array([ 227.5,-597.5, 199])}

a = 143
mirror_diam = 150  # For visualization only. Physical ~25 mm
delta_z = 7
HU_COLORS = {'H1': 'lightpink','H2': 'lightgreen','H3': 'lightblue','H4': 'lightgray'}


# -------------------- Point-source controls --------------------
# USE_POINT_SOURCE   = True                 # if is point source at a finite dist
# LED_DISTANCE_MM    = 9800                 # user-changeable
# LED_DIVERGENCE_DEG = 6.5                  # full divergence angle in degrees

# chief ray hits the XY geometric centre at z=0
_XC = np.mean([H_POSITIONS[k][0] for k in H_POSITIONS])
_YC = np.mean([H_POSITIONS[k][1] for k in H_POSITIONS])
LED_TARGET_POINT = np.array([_XC, _YC, 0.0])

# Helper functions
def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps: raise ValueError("Attempted to normalise a near‑zero vector.")
    return v / n

def _chief_axis(datetime_str, lat, lon, tz_off):
    # azimuth/elevation interface unchanged
    try:
        local = dt.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        local = dt.datetime(2025, 3, 20, 12, 0, 0)
    utc = local - dt.timedelta(hours=tz_off)
    alt, azi = solar_angles(lat, lon, utc)
    return mod_sph_to_cart(alt, azi)

def _project_hits(points, normal, s_hat, L, use_point_source):
    hits = []
    for P in points:
        a1 = _unit(P - L) if use_point_source else s_hat
        d  = refl_ray(a1, normal)
        if abs(d[2]) < 1e-12:
            continue
        t = (RECEIVER_POS[2] - P[2]) / d[2]
        if t > 0:
            hits.append(P + t*d)
    return np.array(hits) if hits else np.empty((0, 3))


def compute_incoming(datetime_str, lat, lon, tz_off):
    return _chief_axis(datetime_str, lat, lon, tz_off)

def _led_position(s_hat, distance_mm):
    # LED lies upstream along the chief axis so its chief ray hits LED_TARGET_POINT at z=0
    return LED_TARGET_POINT - distance_mm * s_hat

def _within_divergence(a1, axis, full_angle_deg):
    if full_angle_deg is None or full_angle_deg <= 0:
        return True
    half = np.deg2rad(0.5 * full_angle_deg)
    return np.dot(_unit(a1), _unit(axis)) >= np.cos(half)

# Public alias
def within_divergence(a1, axis, full_angle_deg):
    return _within_divergence(a1, axis, full_angle_deg)

def sample_disc_points(center, normal, radius, n_radial, n_angular):
    arb = np.array([0,0,1]) if abs(normal[2])<0.9 else np.array([0,1,0])
    u = np.cross(normal, arb); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    pts = [center]
    for i in range(1, n_radial):
        r = radius * (i/(n_radial-1))
        thetas = np.linspace(0, 2*np.pi, n_angular, endpoint=False)
        for th in thetas:
            pts.append(center + r*(np.cos(th)*u + np.sin(th)*v))
    return np.array(pts)

def sample_disc_points_2d(center, normal, radius, spacing):
    arb = np.array([0.0,0.0,1.0])
    if abs(normal[2]) > 0.9: arb = np.array([0.0,1.0,0.0])
    u = np.cross(normal, arb); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    coords = np.arange(-radius, radius + spacing/2, spacing)
    xv, yv = np.meshgrid(coords, coords); xv = xv.ravel(); yv = yv.ravel()
    mask = (xv**2 + yv**2) <= radius**2; xv = xv[mask]; yv = yv[mask]
    return center[None,:] + np.outer(xv, u) + np.outer(yv, v)

def compute_heliostat_data(h_key, tilt_deg, datetime_str, lat, lon, tz_off,
                           use_point_source: bool,
                           led_distance_mm: float,
                           led_divergence_deg: float):
    H = H_POSITIONS[h_key]
    s_hat = _unit(_chief_axis(datetime_str, lat, lon, tz_off))  # chief axis

    # Choose incoming direction used for orientation
    if use_point_source:
        L = _led_position(s_hat, led_distance_mm)
        incoming_for_orientation = _unit(H - L)    # at the heliostat
    else:
        L = None
        incoming_for_orientation = s_hat

    # Receiver direction and base normal
    HR = RECEIVER_POS - H
    r_hat = _unit(HR)
    n_m   = _unit(r_hat - incoming_for_orientation)
    n_mi  = np.array([n_m, n_m])  # two mirrors per heliostat

    # Heliostat local axes
    u_hat = _unit(np.cross(incoming_for_orientation, r_hat))
    v_hat = _unit(np.cross(n_m, u_hat))

    # Mirror centres before tilt and pivots
    mi  = np.array([H - a*u_hat + delta_z*n_mi[0],
                    H + a*u_hat + delta_z*n_mi[1]])
    piv = np.array([H - a*u_hat, H + a*u_hat])
    offset = np.array([mi[i] - piv[i] for i in range(2)])

    # Apply inward tilts
    phi   = np.deg2rad(tilt_deg)
    signs = (+1, -1)
    n_mi_final = np.array([_unit(rodrigues(n_mi[i], v_hat, signs[i]*phi)) for i in range(2)])
    mi_final   = np.array([piv[i] + rodrigues(offset[i], v_hat, signs[i]*phi) for i in range(2)])

    # Reflection directions (unit) for the two mirror centres (no gating)
    refl_dirs = []
    for C, N in zip(mi_final, n_mi_final):
        a1 = _unit(C - L) if use_point_source else s_hat
        d_hat = _unit(refl_ray(a1, N))
        refl_dirs.append(d_hat)
    refl_dirs = np.array(refl_dirs)

    # Rays/segments and endpoints on receiver (keep existing LED-divergence gating)
    endpoints = []
    incoming_segs = []
    reflected_segs = []
    chief_axis = s_hat
    for C, N in zip(mi_final, n_mi_final):
        a1 = _unit(C - L) if use_point_source else s_hat
        if use_point_source and not _within_divergence(a1, chief_axis, led_divergence_deg):
            continue
        d = refl_ray(a1, N)
        if abs(d[2]) < 1e-12:
            continue
        t = (RECEIVER_POS[2] - C[2]) / d[2]
        if t <= 0:
            continue
        E = C + t*d
        endpoints.append(E)
        incoming_segs.append((C - a1 * 1000, C))
        reflected_segs.append((C, E))
    endpoints = np.array(endpoints) if endpoints else np.empty((0,3))

    # Per-mirror surface sampling (geometric outline, no divergence gating)
    filled_all = []
    ring_all = []
    for C, N in zip(mi_final, n_mi_final):
        # (A) dense fill for footprints
        filled_pts = sample_disc_points_2d(center=C, normal=N, radius=12.5, spacing=0.5)
        filled_all.append(_project_hits(filled_pts, N, s_hat, L, use_point_source))

        # (B) sparse ring for wireframe look
        ring_pts = sample_disc_points(center=C, normal=N, radius=12.5, n_radial=2, n_angular=360)
        ring_all.append(_project_hits(ring_pts, N, s_hat, L, use_point_source))

    # concatenate per-mirror arrays
    surface_hits_filled = np.vstack([a for a in filled_all if a.size]) if any(a.size for a in filled_all) else np.empty(
        (0, 3))
    surface_hits_ring = np.vstack([a for a in ring_all if a.size]) if any(a.size for a in ring_all) else np.empty(
        (0, 3))

    # Mesh
    verts, I, J, K = [], [], [], []
    r = mirror_diam/2; n_pts = 50
    for C, N in zip(mi_final, n_mi_final):
        arb = np.array([0,0,1]) if abs(N[2]) < 0.9 else np.array([0,1,0])
        u = np.cross(N, arb); u /= np.linalg.norm(u)
        v = np.cross(N, u)
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        ring = [C + r*(np.cos(t)*u + np.sin(t)*v) for t in angles]
        base = len(verts); verts.append(C); verts.extend(ring)
        I.extend([base]*n_pts)
        J.extend([base + i + 1 for i in range(n_pts)])
        K.extend([base + i + 2 for i in range(n_pts-1)] + [base + 1])
    verts = np.vstack(verts)

    # Normal arrows
    normal_segs = []
    arrow_len = 50
    for start, N in zip([H, *mi_final], [_unit(HR), *n_mi_final]):
        tip = start + N*arrow_len
        normal_segs.append((start, tip))
        arb = np.array([0,0,1]) if abs(N[2])<0.9 else np.array([0,1,0])
        wing_dir = np.cross(N, arb); wing_dir /= np.linalg.norm(wing_dir)
        for s in (+1, -1):
            wing = tip - N*0.2*arrow_len + s*wing_dir*0.1*arrow_len
            normal_segs.append((tip, wing))

    return {
        "mesh": {"verts": verts, "i": I, "j": J, "k": K, "color": HU_COLORS[h_key]},
        "incoming":   incoming_segs,
        "reflected":  reflected_segs,
        "normals":    normal_segs,
        "hits":       endpoints.tolist(),
        "surface_hits":        surface_hits_filled,
        "surface_hits_filled": surface_hits_filled,
        "surface_hits_ring":   surface_hits_ring,
        "mirror_final_positions": mi_final,
        "mirror_target_normals":  n_mi_final,
        "mirror_reflection_dirs": refl_dirs
    }

